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
from ..discovery.etcd_client import EtcdClient

import etcd3
from .constant import *


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, info):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        #don't change this ttl
        self._etcd = EtcdClient(etcd_endpoints, root=job_id)
        self._etcd.init()

        if not self._etcd.set_server_not_exists(service, server, info):
            raise exception.EdlRegisterError()

        self._t_register = threading.Thread(target=self._refresher)

    def _refresher(self):
        while not self._stop.is_set():
            self._etcd_lock.refresh(service, server)
            time.sleep(3)

    def stop(self):
        self._stop.set()
        self._t_register.join()

    def __exit__(self):
        self.stop()


class PodRegister(object):
    def __init__(self, job_env, pod):
        self._stop = threading.Event()
        self._etcd = EtcdClient(job_env.etcd_endpoints, root=job_env.job_id)
        self._etcd.init()

        self._service_name = ETCD_POD_RANK
        self._rank, self._server = self._register_rank(job_env, pod)
        self._t_register = threading.Thread(target=self._refresher)
        self._lock = threading.Lock()
        self._changed = False
        self._pod = pod
        self._job_env = job_env

    def _register_rank(self, job_env, pod, timeout=300):
        rank = -1
        begin = time.time()
        for rank in range(0, job_env.up_limit_nodes):
            server = "{}".format(rank)
            valid = True
            while valid:
                try:
                    pod._rank = rank
                    info = pod.to_json()
                    if not self._etcd.set_server_not_exists(
                            self._service_name, server, info=info, timeout=0):
                        valid = False
                        continue

                    logger.info("register rank:{} on etcd".format(rank))
                    # set rank
                    pod.rank = rank
                    return rank, server
                except (etcd3.exceptions.ConnectionTimeoutError,
                        etcd3.exceptions.ConnectionFailedError
                        ) as e:  # timeout and other
                    if time.time() - begin > timeout:
                        raise EdlRegisterError(
                            "register {} to etcd:{} timeout:{}".format(
                                server, job_env.ectd_endpoints, timeout))
                    time.sleep(1)
                    logger.warning("register to etcd error:{}".format(e))
                    continue

        pod._rank = -1
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

    def __exit__(self):
        self.stop()

    def is_leader(self):
        return self._rank == 0

    def changed(self):
        with self._lock:
            return self._changed

    def complete(self):
        pod.status = PodStatus.COMPLETE
        info = pod.to_json()
        self._etcd.set_server_permanent(self._server_name, self._server, info)
        self.stop()


class PodResourceRegister(Register):
    def __init__(self, etcd_endpoints, job_id, pod):
        """
        /jobid/data_reader/nodes/rank:value
        So the others can find it.
        """
        service = ETCD_POD_RESOURCE
        server = "{}".format(pod.get_id())
        value = pod.to_json()

        super(PodResourceRegister, self).__init__(
            etcd_endpoints=etcd_endpoints,
            job_id=job_id,
            service=service,
            server=server,
            info=value)


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
            etcd_endpoints=etcd_endpoints,
            job_id=job_id,
            service=service,
            server=server,
            info=value)
