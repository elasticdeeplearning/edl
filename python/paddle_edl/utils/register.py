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


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, value):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        self._etcd = EtcdClient(etcd_endpoints, root=job_id, ttl=10)

        if not self._etcd.set_server_not_exists(service_name, server):
            raise exception.EdlRegisterError()

        self._t_register = Threading(self._refresher)

    def _refresher(self):
        while not self._stop.is_set():
            self._etcd_lock.refresh(service_name, server)
            time.sleep(3)

    def stop(self):
        self._stop.set()
        self._t_register.join()


class PodRegister(object):
    def __init__(self, job_env, pod):
        info = self._generate_info(etcd_endpoints, job_env, pod)

        sefl._service_name = "pod"
        self._server = pod._id

        self._register = Register(
            etcd_endpoints=job_env.etcd_endpoints,
            job_id=job_env.job_id,
            service=self._service_name,
            server=self._server,
            info=pod.to_json())

    def stop(self):
        self._register.stop()

    def complete(self):
        info = self._generate_info(
            etcd_endpoints, job_id, pod_id, endpoint, gpus, complete=1)
        self._etcd.set_server_permanent(self._server_name, self._server, info)
        self.stop()


class MasterRegister(object):
    def __init__(self, job_env, pod):
        info = self._generate_info(etcd_endpoints, job_env, pod)

        sefl._service_name = "master"
        self._server = "master"
        self._lock = threading.Lock()
        self._is_master = False
        self._info = pod.to_json()

        self._t_register = Threading(self._refresher)
        self._t_manager = Threading(self._master_manager)
        self._proc = None

    def _refresher(self):
        while not self._stop.is_set():
            try:
                if not self._etcd.set_server_not_exists(self._service_name,
                                                        self._server):
                    raise exception.EdlRegisterError()
            except Exception as e:
                if self._proc is not None:
                    self._terminate_master()
                time.sleep(1)
                continue

            while True:
                try:
                    self._etcd.refresh(self._service_name, self._server)
                    if self._proc is None:
                        self._start_master()
                    else:
                        if self._proc.proc.poll() is not None:  # terminate
                            self._set_master(False)
                            break

                    time.sleep(1)
                except Exception as e:
                    break

    def _set_master(self, is_master):
        with self._lock:
            self._is_master = is_master

    def _start_master(self):
        with self._lock:
            self._proc = utils.start_master()
            self._is_master = True

    def _terminate_master(self):
        with self._lock:
            utils.termniate_master()
            self._proc = None
            self._is_master = False

    def stop(self):
        self._stop.set()
        self._t_register.join()

    def is_master(self):
        with self._lock:
            return self._is_master
