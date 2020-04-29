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


class MasterWatcher(object):
    def __init__(self, etcd_endpoints, job_id):
        self._etcd = EtcdClient(edl_env.etcd_endpoints, root=job_id)
        self._t_watcher = Threading(selt._get_master)
        self._lock = Lock()
        self._master = None

    def _get_master(self):
        while True:
            servers = self._etcd.get_service("master")
            with self._lock:
                if len(servers) == 0:
                    self._master = None
                elif len(servers) == 1:
                    self._master = servers[0][1]
                else:
                    self._master = None
                    logger.fatal(
                        "master must be less than 1, but now:{} server:{}".
                        format(len(servers), servers))

            time.sleep(3)

    def get_master(self):
        with self._lock:
            return self._master


class PodRegister(object):
    def __init__(self, etcd_endpoints, job_id, pod_id, pod_info):
        self._etcd = EtcdClient(etcd_endpoints, root=job_id)
        self._t_register = Threading(self._set_server)
        self._lock = Lock()
        self._pod_id = pod_id
        self._pod_info = pod_info

        self._etcd.set_server(
            service_name="trainer_pod",
            server_name=pod_id,
            info=pod_info,
            ttl=6)

    def _set_server(self):
        with self._lock:
            self._etcd.refresh(
                service_name="trainer_pod", server_name=pod_id, ttl=6)
            time.sleep(3)
