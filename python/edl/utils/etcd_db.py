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
from .pod import Pod
from ..discovery.etcd_client import EtcdClient
from .cluster import Cluster

import etcd3
from .global_vars import *
from .exceptions import EdlBarrierError

import threading
from ..discovery.etcd_client import EtcdClient


class EtcdDB(object):
    # TODO(gongwb): make a connections pool
    def __init__(self, etcd_endpoints, job_id):
        self._lock = threading.Lock()
        self._etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=6)
        self._etcd.init()

    def set_pod_status(self, pod_id, status):
        service = ETCD_POD_STATUS
        server = pod_id
        info = json.dumps({"status": int(status)})
        with self._lock:
            self._etcd.set_server_permanent(service, server, info)

    def get_pods_status(self):
        service = ETCD_POD_STATUS
        with self._lock:
            servers = self._etcd.get_service(service)

        inited = set()
        runing = set()
        succeed = set()
        failed = set()
        for server in servers:
            d = json.loads(server.info)
            status = d["status"]
            if status == int(Status.FAILED):
                failed.add(server.server)
            elif status == int(Status.SUCCEED):
                succeed.add(server.server)
            elif status == int(Status.INITIAL):
                inited.add(server.server)
            elif status == int(Status.RUNNING):
                running.add(server.server)

        return inited, running, succeed, failed

    def set_job_status(self, status):
        service = ETCD_JOB_STATUS
        server = "status"
        info = json.dumps({"status": int(status)})
        with self._lock:
            self._self._etcd.set_server_permanent(service, server, info)

    def set_job_flag(self, flag):
        if flag:
            self.set_job_status(pod.get_id(), Status.SUCCEED)
            logger.info("This job succeeded!")
            return

        logger.fatal("This job meets error!")

    def get_job_status(self):
        service = ETCD_JOB_STATUS
        with self._lock:
            servers = self._etcd.get_service(service)

        assert len(servers) <= 1
        if len(servers) < 1:
            return None

        s = servers[0]
        d = json.loads(s.info)
        return d["status"]

    def wait_following_ranks(self, timeout=60):
        """
        Note: some pod may regist to ranks when other waiting already.
        """
        service = ETCD_POD_RANK
        start = time.time()

        while True:
            with self._lock:
                servers = self._etcd.get_service(service)

            if time.time() - start >= timeout:
                cluster = self.get_rank_cluster()
                raise EdlWaitFollowersReleaseError(
                    "the pods did't release their register:{}".format(
                        cluster.get_pods_ids()))

            if len(servers) > 1:
                time.sleep(2)
                continue

            if len(servers) < 1:
                return True

            pod = Pod()
            pod.from_json(servers[0].info)

            if pod.rank != 0:
                continue

            return True

    def get_resource_pods_dict(self):
        with self._lock:
            servers = self._etcd.get_service(ETCD_POD_RESOURCE)

        pods = {}
        for s in servers:
            p = Pod()
            p.from_json(s.info)
            pods[p.get_id()] = p

        return pods

    def get_pod_leader_id(self):
        with self._lock:
            value = self._etcd.get_value(ETCD_POD_RANK, "0")

        return value

    def get_cluster(self):
        begin = time.time()
        leader_id = EtcdDB.get_pod_leader_id()
        with self._lock:
            value = self._etcd.get_value(ETCD_CLUSTER, leader_id)

        cluster = Cluster()
        cluster.from_json(value)
        if len(cluster.pods) == 0:
            raise EdlGetClusterError("get cluster error")

        return cluster

    def get_pod_leader(self, timeout=15):
        cluster = EtcdDB.get_cluster(timeout)
        return cluster.pods[0]

    def get_data_reader_leader(self):
        raise NotImplementedError()

    def get_current_cluster(self):
        with self._lock:
            value = self._etcd.get_value(ETCD_CLUSTER, ETCD_CLUSTER)

        if value is None:
            return None

        cluster = Cluster()
        cluster.loads(value)
        return cluster

    def set_pod_flag(self, pod_id, flag):
        if not flag:
            EtcdDB.set_pod_status(pod.get_id(), Status.FAILED)
            logger.fatal("local trainers meets error!")
            return

        EtcdDB.set_pod_status(pod.get_id(), Status.SUCCEED)
        logger.info("local trainers succeeded!")

    def get_train_status(self):
        leader_id = self.get_pod_leader_id()
        with self._lock:
            value = self._etcd.get_value(ETCD_TRAIN_STATUS, leader_id)

        if value is None:
            return None

        d = json.load(value)
        return d["status"]

    def set_train_status(self, pod_id, status):
        service = ETCD_TRAIN_STATUS
        server = pod_id
        info = json.dumps({"status": int(status)})
        with self._lock:
            self._etcd.set_server_permanent(service, server, info)

    def wait_resource(self, pod, timeout=15):
        pods = EtdbDB.get_resource_pods_dict()
        if len(pods) == 1:
            if pod.get_id() in pods:
                return True

        if len(pods) == 0:
            return True

        return False


g_etcd_db = None


def get_global_etcd(etcd_endpoints=None, job_id=None):
    global g_etcd_db
    if g_etcd_db is None:
        assert etcd_endpoints is not None and job_id is not None
        g_etcd_db = EtcdDB(etcd_endpoints, job_id)
        return g_etcd_db

    return g_etcd_db
