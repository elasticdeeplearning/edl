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

import enum
from edl.utils import constants
from edl.utils import error_utils


class TrainStatus(enum.IntEnum):
    INITIAL = 0
    RUNNING = 1
    NEARTHEEND = 3
    SUCCEED = 3
    FAILED = 4


@error_utils.handle_errors_until_timeout
def save_to_etcd(etcd, pod_id, status, timeout=30):
    service = constants.ETCD_TRAIN_STATUS
    server = pod_id
    info = json.dumps({"status": int(status)})
    etcd.set_server_permanent(service, server, info)


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, pod_id, timeout=30):
    value = self._etcd.get_value(constants.ETCD_TRAIN_STATUS, pod_id)

    if value is None:
        return None

    d = json.load(value)
    return d["status"]
