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
import uuid

from .utils import logger
from .cluster import Pod, JobStatus
from ..discovery.etcd_client import EtcdClient

import etcd3
from .global_vars import *


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, info):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        #don't change this ttl
        self._etcd = EtcdClient(endpoints=etcd_endpoints, root=job_id)
        self._etcd.init()

        try:
            self._etcd.set_server_not_exists(service, server, info, ttl=10)
        except Exception as e:
            logger.fatal(
                "connect to etcd:{} error:{} service:{} server:{} info:{}".
                format(etcd_endpoints, e, service, server, info))
            raise e
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


class PodRankRegister(object):
    def __init__(self, job_env, pod):
        self._stop = threading.Event()
        self._etcd = EtcdClient(job_env.etcd_endpoints, root=job_env.job_id)
        self._etcd.init()

        self._service_name = ETCD_POD_RANK
        self._rank, self._server = self._register_rank(job_env, pod)
        self._t_register = threading.Thread(target=self._refresher)
        self._lock = threading.Lock()
        self._stopped = False
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
                    # register the leader stage
                    if rank == 0:
                        pod._stage = str(uuid.uuid1())
                    else:
                        pod._stage = None

                    info = pod.to_json()
                    if not self._etcd.set_server_not_exists(
                            self._service_name,
                            server,
                            info=info,
                            timeout=0,
                            ttl=10):
                        valid = False
                        continue

                    logger.info("register rank:{} on etcd key:{}".format(
                        rank,
                        self._etcd.get_full_path(self._service_name, server)))
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

        pod._rank = None
        pod._stage = None
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
            except Exception as e:
                # exit when error ocurred
                break

            # don't change the waited time
            time.sleep(1)

        with self._lock:
            self._rank = None
            self._stopped = True

    def update_stage(self):
        if not self.is_leader():
            return

        with self._lock:
            self._pod._stage = str(uuid.uuid1())
            info = self._pod.to_json()

        self._etcd.refresh(self._service_name, self._server, info=info)

    def stop(self):
        self._stop.set()

        if self._t_register.is_alive():
            self._t_register.join()

        with self._lock:
            self._stopped = True
            self._rank = None

    def __exit__(self):
        self.stop()

    def is_leader(self):
        with self._lock:
            return self._rank == 0

    def complete(self, status):
        with self._lock:
            if status:
                self._pod.status = JobStatus.COMPLETE
            else:
                self._pod.status = JobStatus.ERROR

        info = self._pod.to_json()
        self._etcd.set_server_permanent(self._service_name, self._server, info)
        self.stop()

    def is_stoped(self):
        with self._lock:
            return self._stopped


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


def set_job_complete_flag(flag):
    if flag:
        status = JobStatus.COMPLETE
    else:
        status = JobStatus.ERROR

    etcd, lock = get_etcd()
    service = ETCD_POD_COMPLETE_FLAG
    server = "complete"
    info = json.dumps({"flag": status})
    with self._lock:
        self._etcd.set_server_permanent(service, server, info, ttl=10)


def get_job_complete_flag():
    etcd, lock = get_global_etcd()
    service = ETCD_POD_COMPLETE_FLAG
    with self._lock:
        servers = self._etcd.get_service(service)

    assert len(servers) <= 1
    if len(servers) < 1:
        return None

    s = servers[0]
    d = json.loads(s.info)
    if d["flag"] == JobStatus.ERROR:
        return False
    elif d["flag"] == JobStatus.COMPLETE:
        return True
    else:
        assert False, "can't reach here!"
