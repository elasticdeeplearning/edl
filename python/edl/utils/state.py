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
from edl.discovery.etcd_client import EtcdClient
from edl.utils import constants
from edl.utils import error_utils
from edl.utils import exceptions
from edl.utils import unique_name
from edl.utils.log_utils import logger


class DataCheckpoint(object):
    def __init__(self, reader_name=None, file_list=None, data_checkpoint=None):
        self._reader_name = reader_name
        self._file_list = file_list
        #file_idx_in_file_list:[(record_idx_begin, record_idx_end), ...]
        self._data_checkpoint = data_checkpoint

    def to_json(self):
        pass

    def from_json(self):
        pass


class TrainStatusCheckpoint(object):
    def __init__(self, max_epoch_num):
        self._max_epoch_num = max_epoch_num
        self._epoch_no = None
        self._epochs = {}
        self._status = constants.TrainStatus.INITIAL

    def update_epoch(self, epoch_no, step_num, step_time):
        if epoch_no not in self._epoch:
            self._epochs[epoch_no] = {}

        self._epochs[epoch_no] = {
            "epoch_no": epoch_no,
            "step_num": step_num,
            "step_time": step_time
        }

        left_time = (max_epoch_num - epoch_no) * step_num * step_time
        if left_time > 15 * 60:
            self._status = constants.TrainStatus.RUNNING
        else:
            self._status = constants.TrainStatus.NEARTHEEND

        logger.debug("train status left_time is {} train status is {}".format(
            left_time))

    def to_json(self):
        d = {
            "max_epoch_num": self._max_epoch_num,
            "epoch_no": self._epoch_no,
            "epochs": self._epochs,
            "status": int(self._status),
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)
        self._max_epoch_num = d["max_epoch_num"]
        self._epoch_no = d["epoch_no"]
        self._epochs = d["epochs"]
        self._status = d["status"]

    def __str__(self):
        return self.to_json()


class State(object):
    def __init__(self, total_batch_size, user_defined=None):
        # interface
        self._default = {
            "total_batch_size": total_batch_size,
            "world_size": None,
            "epoch_num": None,
            "epoch_no": None,
            "global_step_no": None,
            "step_no_of_epoch": None,
        }
        self._user_defined = user_defined
        self._adjust_func = []

        # internal
        self._name = unique_name.generator("_edl_state_")
        self._model_path = None
        self._data_checkpoint = DataCheckpoint()
        self._train_status = TrainStatusCheckpoint()

    def register_adjust_function(self, f):
        self._adjust_func.append(f)

    @property
    def world_size(self):
        return self._defaults["world_size"]

    @property
    def epoch_num(self):
        return self._defaults["epoch_num"]

    @property
    def epoch_no(self):
        return self._defaults["epoch_no"]

    @property
    def step_no_of_epoch(self):
        return self._defaults["step_no_of_epoch"]

    @property
    def global_step_no(self):
        return self._defaults["global_step_no"]

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
            etcd.get_full_path(constants.ETCD_DIST_READER, state_name), value))

    s = State()
    s.from_json(value)
    return s

@error_utils.error_utils.handle_errors_until_timeout
def save_to_etcd(etcd_endpoints,
                 job_id,
                 pod_id,
                 name,
                 model_path,
                 data_checkpoint,
                 user_defined,
                 timeout=60):
    etcd = EtcdClient(
        endpoints=etcd_endpoints,
        root=job_id,
        timeout=constants.EDL_CONN_TIMEOUT)
    etcd.init()

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