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
import json
import threading

from . import string_utils
from edl.utils import cluster as cluster_utils
from .log_utils import logger
from .pod import Pod
from ..discovery.etcd_client import EtcdClient
from . import constants
from . import exceptions
from . import state


class EtcdDB(object):
    # TODO(gongwb): make a connections pool
    def __init__(self, etcd_endpoints, job_id):
        self._lock = threading.Lock()
        self._etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=6)
        self._etcd.init()

    def set_pod_status(self, pod_id, status):
        service = constants.ETCD_POD_STATUS
        server = pod_id
        info = json.dumps({"status": int(status)})
        with self._lock:
            self._etcd.set_server_permanent(service, server, info)

    def get_pods_status(self):
        service = constants.ETCD_POD_STATUS
        with self._lock:
            servers = self._etcd.get_service(service)

        inited = set()
        running = set()
        succeed = set()
        failed = set()
        for server in servers:
            d = json.loads(server.info)
            status = d["status"]
            if status == int(constants.Status.FAILED):
                failed.add(server.server)
            elif status == int(constants.Status.SUCCEED):
                succeed.add(server.server)
            elif status == int(constants.Status.INITIAL):
                inited.add(server.server)
            elif status == int(constants.Status.RUNNING):
                running.add(server.server)

        return inited, running, succeed, failed

    def set_job_status(self, status):
        service = constants.ETCD_JOB_STATUS
        server = "status"
        info = json.dumps({"status": int(status)})
        with self._lock:
            self._self._etcd.set_server_permanent(service, server, info)

    def set_job_flag(self, flag):
        if flag:
            self.set_job_status(pod.get_id(), constants.Status.SUCCEED)
            logger.info("This job succeeded!")
            return

        logger.fatal("This job meets error!")

    def get_job_status(self):
        service = constants.ETCD_JOB_STATUS
        with self._lock:
            servers = self._etcd.get_service(service)

        assert len(servers) <= 1
        if len(servers) < 1:
            return None

        s = servers[0]
        d = json.loads(s.info)
        return d["status"]

    def get_resource_pods_dict(self):
        with self._lock:
            servers = self._etcd.get_service(constants.ETCD_POD_RESOURCE)

        pods = {}
        for s in servers:
            p = Pod()
            p.from_json(s.info)
            pods[p.get_id()] = p

        return pods

    def get_pod_leader_id(self):
        with self._lock:
            value = self._etcd.get_value(constants.ETCD_POD_RANK,
                                         constants.ETCD_POD_LEADER)

        if value is None:
            return None

        return string_utils.bytes_to_string(value)

    def get_dist_reader_leader(self):
        leader_id = self.get_pod_leader_id()
        if leader_id is None:
            raise exceptions.EdlTableError("leader_id={}:{}".format(
                self.get_rank_table_key(), leader_id))

        with self._lock:
            value = self._etcd.get_value(constants.ETCD_READER, leader_id)

        if value is None:
            raise exceptions.EdlTableError("leader_id={}:{}".format(
                self.get_reader_table_key(leader_id), cluster))

        reader_leader = state.DistReader()
        reader_leader.from_json(value)
        logger.debug("get reader_leader:".format(reader_leader))
        return reader_leader

    def check_dist_readers(self):
        with self._lock:
            servers = self._etcd.get_service(constants.ETCD_READER)

        if len(servers) <= 0:
            raise exceptions.EdlTableError("table:{} has no readers".format(
                constants.ETCD_READER))

        readers = {}
        for s in servers:
            r = state.DistReader()
            r.from_json(s.value)

            readers[r.key] = r

        cluster = self.get_cluster()
        if cluster is None:
            raise exceptions.EdlTableError("table:{} has no readers".format(
                constants.ETCD_CLUSTER))

        if cluster.get_pods_ids_set() != set(readers.keys()):
            raise exceptions.EdlTableError(
                "reader_ids:{} != cluster_pod_ids:{}".format(reader_ids.keys(
                ), cluster.get_pods_ids_set()))

        logger.debug("get readers:{}".format(readers))
        return readers

    def record_to_dist_reader_table(self, endpoint, reader_name, pod_id):
        r = state.DistReader()
        r._pod_id = pod_id
        r._endpoint = endpoint
        r._name = reader_name

        with self._lock:
            self._etcd.set_server_permanent(constants.ETCD_DIST_READER, pod_id,
                                            r.to_json())

    def get_cluster(self):
        with self._lock:
            value = self._etcd.get_value(constants.ETCD_CLUSTER,
                                         constants.ETCD_CLUSTER)

        if value is None:
            return None

        cluster = cluster_utils.Cluster()
        cluster.from_json(value)
        return cluster

    def get_pod_leader(self):
        leader_id = self.get_pod_leader_id()
        cluster = self.get_cluster()

        if leader_id is None:
            raise exceptions.EdlTableError("leader_id={}:{}".format(
                self.get_rank_table_key(), leader_id))

        if cluster is None:
            raise exceptions.EdlTableError("cluster={}:{}".format(
                self.get_cluster_table_key(), cluster))

        if cluster.pods[0].get_id() != leader_id:
            raise exceptions.EdlLeaderError("{} not equal to {}".format(
                cluster.pods[0].get_id(), leader_id))

        return cluster.pods[0]

    def set_pod_flag(self, pod_id, flag):
        if not flag:
            self.set_pod_status(pod.get_id(), constants.Status.FAILED)
            logger.fatal("local trainers meets error!")
            return

        self.set_pod_status(pod.get_id(), constants.Status.SUCCEED)
        logger.info("local trainers succeeded!")

    def get_train_status(self):
        leader_id = self.get_pod_leader_id()
        if leader_id is None:
            return None

        with self._lock:
            value = self._etcd.get_value(constants.ETCD_TRAIN_STATUS,
                                         leader_id)

        if value is None:
            return None

        d = json.load(value)
        return d["status"]

    def get_train_status_table_key(self, server_name):
        return self._etcd.get_full_path(constants.ETCD_TRAIN_STATUS,
                                        server_name)

    def get_cluster_table_key(self):
        return self._etcd.get_full_path(constants.ETCD_CLUSTER,
                                        constants.ETCD_CLUSTER)

    def get_rank_table_key(self):
        return self._etcd.get_full_path(constants.ETCD_POD_RANK,
                                        constants.ETCD_POD_LEADER)

    def get_reader_table_key(self, pod_id):
        return self._etcd.get_full_path(constants.ETCD_READER, pod_id)

    def set_train_status(self, pod_id, status):
        service = constants.ETCD_TRAIN_STATUS
        server = pod_id
        info = json.dumps({"status": int(status)})
        with self._lock:
            self._etcd.set_server_permanent(service, server, info)

    def wait_resource(self, pod, timeout=15):
        pods = EtcdDB.get_resource_pods_dict()
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
