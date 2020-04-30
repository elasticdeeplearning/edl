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


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, value):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        self._etcd = EtcdClient(etcd_endpoints, root=job_id, ttl=10)

        if not self._etcd.set_server_not_exists(service_name, server):
            raise exception.CanNotRegister()

        self._t_register = Threading(self._refresher)

    def _refresher(self):
        while not self._stop.is_set():
            self._etcd_lock.refresh(service_name, server)
            time.sleep(3)

    def stop(self):
        self._stop.set()
        self._t_register.join()


class LauncherRegister(object):
    def __init__(self, etcd_endpoints, job_id, pod_id, info):
        service_name = "Launcher"
        server = pod_id

        self._register = Register(
            etcd_endpoints,
            job_id=job_id,
            service=service_name,
            server=server,
            info=info)


class TrainerRegister(object):
    def __init__(self, etcd_endpoints, job_id, pod_id, rank_of_pod, info):
        service_name = "Trainer"
        server = pod_id + str(rank_of_pod)

        self._register = Register(
            etcd_endpoints,
            job_id=job_id,
            service=service_name,
            server=server,
            info=info)


class DataServerRegister(object):
    def __init__(self, etcd_endpoints, job_id, pod_id, rank_of_pod, info):
        service_name = "DataServer"
        server = pod_id + str(trainer_rank)
        value = trainer_info

        self._register = Register(
            etcd_endpoints,
            job_id=job_id,
            service=service_name,
            server=server,
            info=info)
