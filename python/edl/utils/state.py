# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import six
from edl.utils import constants
from edl.utils import error_utils
from edl.utils import exceptions
from edl.utils import json_serializable
from edl.utils import string_utils
from edl.utils import train_status as edl_train_status
from edl.utils import unique_name


class DataCheckpoint(json_serializable.Serializable):
    def __init__(self, reader_name=None, file_list=None, processed_data=None):
        self.reader_name = reader_name
        self.file_list = file_list
        # dict, file_idx_in_file_list => \
        # [(record_idx_begin, record_idx_end), ...]
        self.processed_data = processed_data


class EpochAttr(json_serializable.Serializable):
    def __init__(self):
        self.epoch_no = None
        self.world_size = None
        self.step_num = None
        self.avg_step_time = None
        self.step_no_of_epoch = None


def _load_dict_of_cls_from_json(json_str, cls):
    d = json.loads(json_str)

    ret = {}
    for k, v in six.iteritems(d):
        ret[int(k)] = cls().from_json(v)

    return ret


def _dump_dict_to_json(d):
    ret = {}
    for k, v in six.iteritems(d):
        ret[int(k)] = v.to_json()

    return json.dumps(ret)


class TrainStatus(json_serializable.Serializable):
    def __init__(self):
        self._epoch_no = None  # current
        self.global_step_no = None  # current

        self._epochs = {}  # epoch_no => EpochAttr
        self.status = edl_train_status.TrainStatus.INITIAL

    def to_json(self):
        d = {
            "_epoch_no": self._epoch_no,
            "global_step_no": int(self.global_step_no),
            "_epochs": _dump_dict_to_json(self._epochs),
            "status": int(self.status),
        }

        return json.dumps(d)

    def from_json(self, json_str):
        d = json.loads(json_str)
        self._epoch_no = d["_epoch_no"]
        self.global_step_no = d["global_step_no"]
        self.status = d["status"]

        print("d[epochs]", d["_epochs"])
        self._epochs = _load_dict_of_cls_from_json(d["_epochs"], EpochAttr)

    @property
    def epoch_no(self):
        return self._epoch_no

    @epoch_no.setter
    def epoch_no(self, epoch_no):
        assert epoch_no >= 0
        if epoch_no not in self._epochs:
            self._epochs[epoch_no] = {}
        self._epoch_no = epoch_no

    def get_epoch_attr(self, epoch_no):
        if epoch_no not in self._epochs:
            return None
        return self._epochs[epoch_no]

    def update_epoch_attr(self, epoch_no, epoch_attr):
        self._epochs[epoch_no] = epoch_attr

    def get_current_epoch_attr(self):
        return self.get_epoch_attr(self._epoch_no)

    def update_current_epoch_attr(self, epoch_attr):
        return self.update_epoch_attr(self._epoch_no, epoch_attr)


class State(json_serializable.Serializable):
    def __init__(self, total_batch_size, user_defined=None):
        # interface
        self._default = {
            "total_batch_size": total_batch_size,  # user inputs
        }
        self._user_defined = user_defined
        self._adjust_func = []

        # internal
        self._name = unique_name.generator("_edl_state_")
        self._model_path = None
        self._data_checkpoint = DataCheckpoint()
        self._train_status = TrainStatus()

    def from_json(self, json_str):
        d = json.loads(json_str)

        self._default = d["_default"]
        if self._user_defined is not None and d["_user_defined"] is not None:
            self._user_defined.from_json(d["_user_defined"])

        self._name = d["_name"]
        self._model_path = d["_model_path"]
        self._data_checkpoint.from_json(d["_data_checkpoint"])
        self._train_status.from_json(d["_train_status"])
        return d

    def register_adjust_function(self, f):
        self._adjust_func.append(f)

    @property
    def name(self):
        return self._name

    @property
    def epoch_no(self):
        return self._train_status.epoch_no

    @property
    def step_no_of_epoch(self):
        return self._train_status.get_current_epoch_attr().step_no_of_epoch

    @property
    def global_step_no(self):
        return self._train_status.global_step_no

    @property
    def total_batch_size(self):
        return self._default["total_batch_size"]

    @total_batch_size.setter
    def total_batch_size(self, size):
        self._default["total_batch_size"] = size


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, state_name, user_defined=None, timeout=60):
    value = etcd.get_value(constants.ETCD_STATE, state_name)

    if value is None:
        raise exceptions.EdlTableError(
            "key:value = {}:{}".format(
                etcd.get_full_path(constants.ETCD_READER, state_name), value
            )
        )

    state = State(total_batch_size=None, user_defined=user_defined)
    state.from_json(string_utils.bytes_to_string(value))
    return state


@error_utils.handle_errors_until_timeout
def save_to_etcd(etcd, pod_id, state, timeout=60):
    leader_key = etcd.get_full_path(constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER)
    state_key = etcd.get_full_path(constants.ETCD_STATE, state.name)

    etcd = etcd._etcd
    status, _ = etcd.transaction(
        compare=[etcd.transactions.value(leader_key) == pod_id,],  # noqa: E231
        success=[etcd.transactions.put(state_key, state.to_json()),],  # noqa: E231
        failure=[],
    )

    message = "pod_id:{} save_data_checkpoint status:{}".format(pod_id, status)
    if not status:
        raise exceptions.EdlEtcdIOError(message)


class PaddleState(State):
    def __init__(
        self,
        total_batch_size,
        user_defined=None,
        optimizer=None,
        exe=None,
        program=None,
    ):
        super(PaddleState, self).__init__(
            total_batch_size=total_batch_size, user_defined=user_defined
        )
        self._exe = exe
        self._program = program
        self._optimizer = optimizer
