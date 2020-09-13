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
#ETCD_DIST_READER = "dist_reader"
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