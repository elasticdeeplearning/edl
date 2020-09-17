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
from edl.utils import cluster as cluster_utils

from . import constants
from . import exceptions
from . import state
from .log_utils import logger
from .pod import Pod
from ..discovery.etcd_client import EtcdClient


class EtcdDB(object):
    # TODO(gongwb): make a connections pool
    def __init__(self, etcd_endpoints, job_id):
        self._lock = threading.Lock()
        self._etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=6)
        self._etcd.init()

    def get_resource_pods_dict(self):
        with self._lock:
            servers = self._etcd.get_service(constants.ETCD_POD_RESOURCE)

        pods = {}
        for s in servers:
            p = Pod()
            p.from_json(s.info)
            pods[p.get_id()] = p

        return pods



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



    def set_pod_flag(self, pod_id, flag):
        if not flag:
            self.set_pod_status(pod.get_id(), constants.Status.FAILED)
            logger.fatal("local trainers meets error!")
            return

        self.set_pod_status(pod.get_id(), constants.Status.SUCCEED)
        logger.info("local trainers succeeded!")





    def wait_resource(self, pod, timeout=15):
        pods = EtcdDB.get_resource_pods_dict()
        if len(pods) == 1:
            if pod.get_id() in pods:
                return True

        if len(pods) == 0:
            return True

        return False


g_etcd = None


def get_global_etcd(etcd_endpoints=None, job_id=None):
    global g_etcd
    if g_etcd is None:
        assert etcd_endpoints is not None and job_id is not None
        return g_etcd

    return g_etcd
