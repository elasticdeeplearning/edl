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
from edl.utils import train_status as edl_train_status
from edl.utils import unique_name
from edl.utils.log_utils import logger


def _compare_two_dict(dict1, dict2):
    if len(dict1) != dict2:
        return False

    for k,v in six.iteritems(dict1):
        if k not in dict1:
            return False

        if isinstance(v, dict):
            if not _compare_two_dict(v, dict2[k]):
                return False
        else:
            if v != dict2[k]:
                return False

    return True

class DataCheckpoint(object):
    def __init__(self, reader_name=None,
                 file_list=None, processed_data=None):
        self._reader_name = reader_name
        self._file_list = file_list
        #dict, file_idx_in_file_list=>[(record_idx_begin, record_idx_end), ...]
        self._processed_data = processed_data

    def to_json(self):
        d = {
            "reader_name": self._reader_name,
            "file_list": self._file_list,
            "processed_data": json.dumps(self._processed_data),
        }
        return json.dumps(d)

    def from_json(self, json_str):
        d = json.loads(json_str)
        self._reader_name = d["reader_name"]
        self._file_list = d["file_list"]
        self._processed_data = json.loads(d["processed_data"])
        return self

    def __eq__(self, other):
        if self._reader_name != other._reader_name or \
            self._file_list != other._file_list:
            return False

        return _compare_two_dict(self._processed_data, other._processed_data)

    def __ne__(self, other):
        return not self.__eq__(other)


class TrainStatus(object):
    def __init__(self):
        self._epoch_no = None # current
        self._step_no_of_epoch = None # current
        self._global_step_no = None # current

        self._epochs = {} # epoch_no => epoch attribute
        self._status = edl_train_status.TrainStatus.INITIAL

    @property
    def epoch_no(self):
        return self._epoch_no

    @epoch_no.setter
    def epoch_no(self, epoch_no):
        assert epoch_no >= 0
        if epoch_no not in self._epoch:
            self._epochs[epoch_no] = {}
        self._epoch_no = epoch_no

    @world_size.setter
    def world_size(self, world_size):
        self._epochs[self._epoch_no]["world_size"] = world_size

    @step_no_of_epoch.setter
    def step_no_of_epoch(self, step_no):
        self._epochs[self._epoch_no]["step_no_of_epoch"] = step_no

    @step_num.setter
    def step_num(self, step_num):
        self._epochs[self._epoch_no]["step_num"] = step_num

    @step_time.setter
    def step_time(self, step_time):
        self._epochs[self._epoch_no]["step_time"] = step_time

    def update_status(self):
        assert self._epoch_no >= 0
        if epoch_no not in self._epoch:
            self._epochs[epoch_no] = {}

        self._epochs[epoch_no] = {
            "epoch_no": epoch_no,
            "step_num": step_num,
            "step_time": step_time,
            "world_size": world_size,
        }

        left_time = (self._max_epoch_num - self._epoch_no) * self.step_num * step_time
        if left_time > 15 * 60:
            self._status = edl_train_status.TrainStatus.RUNNING
        else:
            self._status = edl_train_status.TrainStatus.NEARTHEEND

        logger.debug("train status left_time is {} train status is {}".format(
            left_time, self._status))

    def to_json(self):
        d = {
            "epoch_no": self._epoch_no,
            "step_no_of_epoch": self._step_no_of_epoch,
            "global_step_no": int(self._global_step_no),
            "epochs": json.dumps(self._epochs),
            "status": int(self._status),
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)

        self._epoch_no = d["epoch_no"]
        self._step_no_of_epoch = d["step_no_of_epoch"]
        self._global_step_no = d["global_step_no"]
        self._epochs = json.loads(d["epochs"])
        self._status = d["status"]

    def __str__(self):
        return self.to_json()

    def __eq__(self, other):
        if self._epoch_no != other._epoch_no or \
            self._step_no_of_epoch != other._step_no_of_epoch or \
            self._global_step_no != other._global_step_no or \
            self._status != other._status:
            return False

        return _compare_two_dict(self._epochs, other._epochs)

    def __ne__(self, other):
        return not self.__eq__(other)

class State(object):
    def __init__(self, total_batch_size, user_defined=None):
        # interface
        self._default = {
            "total_batch_size": total_batch_size, # user inputs
        }
        self._user_defined = user_defined
        self._adjust_func = []

        # internal
        self._name = unique_name.generator("_edl_state_")
        self._model_path = None
        self._data_checkpoint = DataCheckpoint()
        self._train_status = TrainStatus()

    def register_adjust_function(self, f):
        self._adjust_func.append(f)

    @property
    def world_size(self):
        return self._train_status._epochs[]

    @property
    def epoch_no(self):
        return self._train_status._epoch_no

    @property
    def step_no_of_epoch(self):
        return self._train_status._step_no_of_epoch

    @property
    def global_step_no(self):
        return self._train_status._global_step_no

    @property
    def total_batch_size(self):
        return self._defaults["total_batch_size"]

    @total_batch_size.setter
    def total_batch_size(self, size):
        self._defaults["total_batch_size"] = size

    def to_json(self):
        d = {
            "default": json.to_json(self._default),
            "user_defined": self._user_defined.to_json()
            if self._user_defined else None,
            "name": self._name,
            "model_path": self._model_path,
            "data_checkpoint": self._data_checkpoint.to_json(),
            "train_status": self._train_status.to_json(),
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)

        self._defaults = d["default"]
        if self._user_defined is not None and d["user_defined"] is not None:
            self._user_defined.from_json(d["user_defined"])

        self._name = d["name"]
        self._model_path = d["model_path"]
        self._data_checkpoint.from_json(d["data_checkpoint"])
        self._train_status.from_json(d["train_status"])
        return d


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, state_name, timeout=60):
    value = etcd.get_value(constants.ETCD_STATE, state_name)

    if value is None:
        raise exceptions.EdlTableError("key:value = {}:{}".format(
            etcd.get_full_path(constants.ETCD_READER, state_name), value))

    s = State()
    s.from_json(value)
    return s


@error_utils.error_utils.handle_errors_until_timeout
def save_to_etcd(etcd, state, timeout=60):
    s = State()
    s._name = name
    s._data_checkpoint = data_checkpoint
    s._model_path = model_path
    s._user_defined = user_defined

    leader_key = etcd.get_full_path(constants.ETCD_POD_RANK,
                                    constants.ETCD_POD_LEADER)
    state_key = etcd.get_full_path(constants.ETCD_STATE, name)

    etcd = etcd._etcd
    status, _ = etcd.transaction(
        compare=[etcd.transactions.value(leader_key) == pod_id, ],
        success=[etcd.transactions.put(state_key, s.to_json()), ],
        failure=[])

    message = "pod_id:{} save_data_checkpoint status:{}".format(pod_id, status)
    if not status:
        raise exceptions.EdlEtcdIOError(message)

    return


class PaddleState(State):
    def __init__(self,
                 total_batch_size,
                 user_defined=None,
                 optimizer=None,
                 exe=None,
                 program=None):
        super(PaddleState, self).__init__(
            total_batch_size=total_batch_size, user_defined=user_defined)
        self._exe = exe
        self._program = program
        self._optimizer = optimizer
