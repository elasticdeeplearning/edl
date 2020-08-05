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
from paddle_edl.discovery.etcd_client import EtcdClient
import json


class Master(object):
    def __init__(self, endpoint):
        self.endpoint = endpoint


class MasterWatcher(object):
    def __init__(self, etcd_endpoints, job_id):
        self._etcd = EtcdClient(edl_env.etcd_endpoints, root=job_id)
        self._job_id = job_id
        self._master = None

        self._lock = Lock()
        # wait to get master
        self._get_master()

        self._t_watcher = Threading(selt._get_master)
        self._t_watcher.start()

    def _get_master(self):
        begin = time.time()
        while True:
            v, _ = self._etcd.get_key("/{}/master/meta".format(self._job_id))
            with self._lock:
                d = json.loads(v)
                if self._master is None:
                    self._master = Master()
                self._master.endpoint = d['endpoint']
                self._master.job_stage = d['job_stage']

            time.sleep(1)

    def get_master(self):
        with self._lock:
            return self._master
