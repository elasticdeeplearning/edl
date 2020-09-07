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

from enum import IntEnum

ETCD_POD_RESOURCE = "pod_resource"
ETCD_POD_RANK = "rank"
ETCD_POD_STATUS = "pod_status"
ETCD_JOB_STATUS = "job_status"
ETCD_TRAIN_STATUS = "train_status"
ETCD_CLUSTER = "cluster"
ETCD_DIST_READER = "dist_reader"
ETCD_STATE = "state"
ETCD_POD_LEADER = "0"

ETCD_CONN_TIMEOUT = 6
ETCD_TTL = 15


class Status(IntEnum):
    INITIAL = 0
    RUNNING = 1
    PENDING = 2
    SUCCEED = 3
    FAILED = 4

    @staticmethod
    def bool_to_status(b):
        if b:
            return Status.SUCCEED

        return Status.FAILED


class TrainStatus(IntEnum):
    INITIAL = 0
    RUNNING = 1
    NEARTHEEND = 3
    SUCCEED = 3
    FAILED = 4


class DistReader(object):
    def __init__(self, pod_id, name, endpoint):
        self._pod_id = pod_id
        self._name = name
        self._endpoint = endpoint

    def to_json(self):
        d = {
            "pod_id": self._pod_id,
            "endpoint": self._endpoint,
            "name": self._name,
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)
        self._pod_id = d["pod_id"]
        self._endpoint = d["endpoint"]
        self._name = d["name"]

    def __str_(self):
        return self._to_json()


class DataCheckpoint(object):
    def __init__(self, reader_name, file_list, data_checkpoint):
        self._reader_name = reader_name
        self._file_list = file_list
        self._data_checkpoint = data_checkpoint


class TrainStatusCheckpoint(object):
    def __init__(self, max_epoch_num):
        self._max_epoch_num = max_epoch_num
        self._epoch_no = None
        self._epochs = {}
        self._status = TrainStatus.INITIAL

    def update_epoch(self, epoch_no, step_num, step_time):
        if epoch_no not in self._epoch:
            self._epochs[epoch_no] = {}
        epoch = self._epochs[epoch_no]
        epoch = {
            "epoch_no": epoch_no,
            "step_num": step_num,
            "step_time": step_time
        }

        left_time = (max_epoch_num - epoch_no) * step_num * step_time
        if left_time > 15 * 60:
            self._status = TrainStatus.RUNNING
        else:
            self._status = TrainStatus.NEARTHEEND

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
