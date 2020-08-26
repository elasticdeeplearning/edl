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
import copy

from .utils import logger
from .pod import Pod, JobStatus
from ..discovery.etcd_client import EtcdClient

import etcd3
from .global_vars import *


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, info):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        self._etcd = None
        self._t_register = None

        self._etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=6)
        self._etcd.init()

        try:
            self._etcd.set_server_not_exists(service, server, info, ttl=15)
            logger.info("register pod:{} in resource:{}".format(
                info, self._etcd.get_full_path(service, server)))
        except Exception as e:
            logger.fatal(
                "connect to etcd:{} error:{} service:{} server:{} info:{}".
                format(etcd_endpoints, e, service, server, info))
            raise e

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _refresher(self):
        while not self._stop.is_set():
            try:
                self._etcd.refresh(self._service, self._server)
                time.sleep(2)
            except Exception as e:
                logger.fatal("register meet error and exit! error:".format(e))
                self._stopped = True
                break

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_register:
                self._t_register.join()
                self._t_register = None

    def is_stopped(self):
        with self._lock:
            return self._t_register == None

    def __exit__(self):
        self.stop()


class GenerateCluster(object):
    def __init__(self):
        pass

    def start(self):
        """
        self._t_make_cluster = threading.Thread(make_cluster)
        self._t_make_cluster.start()
        """
        pass

    def stop(self):
        pass

    def __exit__(self):
        self._stop()


class LeaderRegister(object):
    def __init__(self, job_env, pod):
        self._job_env = job_env
        self._leader = Leader()
        self._is_leader = False
        self._leader._pod_id = pod.get_id()
        self._generate_cluster = None

        self._stop = threading.Event()
        self._service_name = ETCD_POD_RANK
        self._lock = threading.Lock()

        self._etcd = None
        self._t_register = None

        # assign value
        self._etcd = EtcdClient(
            self._job_env.etcd_endpoints, root=self._job_env.job_id, timeout=6)
        self._etcd.init()

        self._seize_leader(job_env, pod.to_json())

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _seize_leader(self, timeout=6):
        begin = time.time()
        server = "0"
        self._leader._stage = str(uuid.uuid1())

        if not self._etcd.set_server_not_exists(
                self._service_name,
                server,
                info=self._leader.to_json(),
                timeout=6,
                ttl=15):
            logger.debug("register rank:{} on etcd key:{} error".format(
                rank, self._etcd.get_full_path(self._service_name, server)))

            with self._lock:
                self._is_leader = False

            return False

        self._generate_cluster = GenerateCluster()
        self._generate_cluster.start()
        with self._lock:
            self._is_leader = True
        logger.info("register rank:{} on etcd key:{}".format(
            rank, self._etcd.get_full_path(self._service_name, server)))
        return True

    def is_leader(self):
        with self._lock:
            return self._is_leader

    def _refresh(self):
        try:
            self._etcd.refresh(self._service_name, self._server)
            return True
        except Exception as e:
            logger.warning("refresh error:{}".format(e))
            with self._lock:
                self._is_leader = False
        return False

    def _refresher(self):
        while not self._stop.is_set():
            with self._lock:
                is_leader = self._is_leader

            try:
                if is_leader:
                    self._refresh()
                else:
                    self._seize_leader()
            except Exception as e:
                # exit when error ocurred
                break

            time.sleep(1)

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_register:
                self._t_register.join()
                self._t_register = None

        logger.info("pod_register stopped")

    def __exit__(self):
        self.stop()

    def is_stopped(self):
        with self._lock:
            return self._t_register == None


class PodResourceRegister(Register):
    """
    Registe pod resource under  /jobid/data_reader/nodes/resource
    If it stop, it means the pods disappear.
    """

    def __init__(self, job_env, pod):
        service = ETCD_POD_RESOURCE
        server = "{}".format(pod.get_id())
        value = pod.to_json()

        super(PodResourceRegister, self).__init__(
            etcd_endpoints=job_env.etcd_endpoints,
            job_id=job_env.job_id,
            service=service,
            server=server,
            info=value)

    def stop(self):
        super(PodResourceRegister, self).stop()
        logger.info("pod resource register stopped!")


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
