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

from threading import Lock, Thread
import time
from utils import logger
from paddle_edl.utils.discovery.etcd_client import EtcdClient
import json


class MasterWatcher(object):
    def __init__(self, etcd_endpoints, job_id, timeout=60):
        self._etcd = EtcdClient(edl_env.etcd_endpoints, root=job_id)
        self._t_watcher = Threading(selt._get_master)
        self._lock = Lock()
        self._master = None
        self._job_id = job_id

    def _get_master(self):
        begin = time.time()
        while True:
            v, _ = self._etcd.get_key("/{}/master/addr".format(self._job_id))
            with self._lock:
                self._master = v
            time.sleep(3)

    def get_master(self):
        with self._lock:
            return self._master
