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
from .pod import Pod, JobStatus
from ..discovery.etcd_client import EtcdClient
from .cluster import Cluster

import etcd3
from .global_vars import *


class EtcdDB(object):
    @staticmethod
    def set_pod_complete_flag(flag, pod_id):
        """
        Set job flags: COMPLETE or ERROR
        """
        if flag:
            status = JobStatus.COMPLETE
        else:
            status = JobStatus.ERROR

        etcd, lock = get_global_etcd()
        service = ETCD_POD_STATUS
        server = pod_id
        info = json.dumps({"flag": int(status)})
        with lock:
            etcd.set_server_permanent(service, server, info)

    @staticmethod
    def get_pods_complete_flag():
        """
        Get succeed pods and failed pods
        """
        etcd, lock = get_global_etcd()
        service = ETCD_POD_STATUS
        with lock:
            servers = etcd.get_service(service)

        succeed = set()
        failed = set()
        for server in servers:
            d = json.loads(server.info)
            if d["flag"] == int(JobStatus.ERROR):
                failed.add(server.server)
            elif d["flag"] == int(JobStatus.COMPLETE):
                succeed.add(server.server)

        return succeed, failed

    @staticmethod
    def wait_following_ranks(timeout=60):
        """
        Note: some pod may regist to ranks when other waiting already.
        """
        etcd, lock = get_global_etcd()
        service = ETCD_POD_RANK
        start = time.time()

        while True:
            with lock:
                servers = etcd.get_service(service)

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

    @staticmethod
    def set_job_complete_flag(flag):
        if flag:
            status = JobStatus.COMPLETE
        else:
            status = JobStatus.ERROR

        etcd, lock = get_global_etcd()
        service = ETCD_JOB_STATUS
        server = "complete"
        info = json.dumps({"flag": int(status)})
        with lock:
            etcd.set_server_permanent(service, server, info)

    @staticmethod
    def get_job_complete_flag():
        etcd, lock = get_global_etcd()
        service = ETCD_JOB_STATUS
        with lock:
            servers = etcd.get_service(service)

        assert len(servers) <= 1
        if len(servers) < 1:
            return None

        s = servers[0]
        d = json.loads(s.info)
        if d["flag"] == int(JobStatus.ERROR):
            return False
        elif d["flag"] == int(JobStatus.COMPLETE):
            return True
        else:
            assert False, "can't reach here!"

    @staticmethod
    def get_resource_pod_ids_set():
        etcd, lock = get_global_etcd()
        with lock:
            pod_resource_servers = etcd.get_service(ETCD_POD_RESOURCE)

        ids = set()
        for m in pod_resource_servers:
            p = Pod()
            p.from_json(m.info)
            ids.add(p.get_id())

        return ids

    @staticmethod
    def get_rank_cluster():
        etcd, lock = get_global_etcd()
        cluster = Cluster()
        servers = etcd.get_service(ETCD_POD_RANK)
        ranks = {}
        for s in servers:
            ranks[int(s.server)] = s.info
        cluster.from_rank_dict(ranks)
        return cluster

    @staticmethod
    def get_pod_leader():
        etcd, lock = get_global_etcd()
        with lock:
            value, _, _, _, _, = etcd._get_server(ETCD_POD_RANK, "0")

        leader = Pod()
        leader.from_json(value)
        return leader

    @staticmethod
    def get_data_reader_leader():
        raise NotImplementedError()

    @staticmethod
    def get_diff_pods_flag(cluster):
        """
        return succeeded and failed pods in old_cluster
        """
        all_succeed, all_failed = EtcdDB.get_pods_complete_flag()
        resource = EtcdDB.get_resource_pod_ids_set()
        init = cluster.get_pods_ids_set()
        now = EtcdDB.get_rank_cluster().get_pods_ids_set()

        added = now - init
        #diff = init.symmetric_difference(now)
        succeeded = init & all_succeed
        #disappeared = init - now - (diff & failed) - succeeded
        failed = init - now - succeeded

        return succeeded, failed, added
