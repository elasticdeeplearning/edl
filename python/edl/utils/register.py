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
import threading
import time
import json

from .utils import logger
from .cluster import Pod, PodStatus
from paddle_edl.discovery.etcd_client import EtcdClient


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, value):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        #don't change this ttl
        self._etcd = EtcdClient(etcd_endpoints, root=job_id, ttl=10)

        if not self._etcd.set_server_not_exists(service_name, server):
            raise exception.EdlRegisterError()

        self._t_register = threading.Thread(self._refresher)

    def _refresher(self):
        while not self._stop.is_set():
            self._etcd_lock.refresh(service_name, server)
            time.sleep(3)

    def stop(self):
        self._stop.set()
        self._t_register.join()


class PodRegister(object):
    def __init__(self, job_env, pod):
        info = pod.to_json()

        self._stop = threading.Event()
        self._etcd = EtcdClient(etcd_endpoints, root=job_id, ttl=10)

        sefl._service_name = "pod"
        self._rank, self._server = self._register_rank(job_env, pod)
        self._t_register = threading.Thread(self._refresher)
        self._lock = threading.Lock()
        self._changed = False
        self._pod = pod
        self._job_env = job_env

    def _register_rank(self, job_env, pod, timeout=300):
        rank = -1
        for rank in range(0, job_env.up_limit_nodes):
            server = "{}".format(rank)
            valid = True
            while valid:
                try:
                    pod.set_id(rank)
                    info = pod.to_json()
                    if not self._etcd.set_server_not_exists(
                            self._service_name, sever, info=self._info,
                            timeout=0):
                        valid = False
                        continue
                    else:
                        logger.info("register rank:{} from etcd".format(rank))
                        return rank, server
                except (etcd3.ConnectionTimeoutError,
                        ConnectionFailedError) as e:  # timeout and other
                    if time.time() - begin > timeout:
                        raise EdlRegisterError(
                            "register {} to etcd:{} timeout:{}".format(
                                server, job_env.ectd_endpoints, timeout))
                    time.sleep(1)
                    continue

        raise EdlRegisterError(
            "register {} to etcd:{} but can't find valid rank:{}".format(
                server, job_env.ectd_endpoints, rank))

    @property
    def rank(self):
        return self._rank

    def _refresher(self):
        while not self._stop.is_set():
            try:
                self._etcd.refresh(self._service_name, self._server)
                # don't change the waited time
                time.sleep(1)
            except Exception as e:
                with self._lock:
                    self._changed = True
                break

    def stop(self):
        self._stop.set()
        self._t_register.join()

    def is_master(self):
        return self._rank == 0

    def changed(self):
        with self._lock:
            return self._changed

    def complete(self):
        pod.status = PodStatus.COMPLETE
        info = pod.to_json()
        self._etcd.set_server_permanent(self._server_name, self._server, info)
        self.stop()


class DataReaderRegister(Register):
    def __init__(self, etcd_endoints, job_id, rank, reader):
        """
        /jobid/data_reader/nodes/rank:value
        So the others can find it.
        """
        service = "data_reader"
        sever = "{}".format(rank)
        value = {
            "id": reader.get_id(),
            "name": reader.name,
            "endpoint": reader.endpoint,
            "rank": rank
        }

        super(DataReaderRegister, self).__init__(
            etcd_endponts=etcd_endpoints,
            job_id=job_id,
            service=service,
            server=server,
            value=value)
