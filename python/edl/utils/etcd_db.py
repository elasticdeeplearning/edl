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
from enum import IntEnum

from .utils import logger
from .pod import Pod, JobStatus
from ..discovery.etcd_client import EtcdClient
from .cluster import Cluster

import etcd3
from .global_vars import *
from .exceptions import EdlBarrierError


class TrainStatus(IntEnum):
    INITIAL = 0
    RUNNING = 1
    NEARTHEEND = 3
    COMPLETE = 3
    ERROR = 4


class EtcdDB(object):
    @staticmethod
    def set_pod_status(pod_id, status):
        etcd, lock = get_global_etcd()
        service = ETCD_POD_STATUS
        server = pod_id
        info = json.dumps({"status": int(status)})
        with lock:
            etcd.set_server_permanent(service, server, info)

    @staticmethod
    def get_pods_status():
        etcd, lock = get_global_etcd()
        service = ETCD_POD_STATUS
        with lock:
            servers = etcd.get_service(service)

        inited = set()
        runing = set()
        succeed = set()
        failed = set()
        for server in servers:
            d = json.loads(server.info)
            status = d["status"]
            if status == int(JobStatus.ERROR):
                failed.add(server.server)
            elif status == int(JobStatus.COMPLETE):
                succeed.add(server.server)
            elif status == int(JobStatus.INITIAL):
                inited.add(server.server)
            elif status == int(JobStatus.RUNNING):
                running.add(server.server)

        return inited, running, succeed, failed

    @staticmethod
    def set_job_status(status):
        etcd, lock = get_global_etcd()
        service = ETCD_JOB_STATUS
        server = "status"
        info = json.dumps({"status": int(status)})
        with lock:
            etcd.set_server_permanent(service, server, info)

    @staticmethod
    def set_job_flag(flag):
        if flag:
            EtcdDB.set_job_status(pod.get_id(), JobStatus.COMPLETE)
            logger.info("This job succeeded!")
            return

        logger.fatal("This job meets error!")

    @staticmethod
    def get_job_status():
        etcd, lock = get_global_etcd()
        service = ETCD_JOB_STATUS
        with lock:
            servers = etcd.get_service(service)

        assert len(servers) <= 1
        if len(servers) < 1:
            return None

        s = servers[0]
        d = json.loads(s.info)
        return d["status"]

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
    def get_resource_pods_ids_set():
        etcd, lock = get_global_etcd()
        with lock:
            servers = etcd.get_service(ETCD_POD_RESOURCE)

        ids = set()
        for s in servers:
            ids.add(s.server)

        return ids

    @staticmethod
    def get_pod_leader_id():
        etcd, lock = get_global_etcd()
        with lock:
            value, _, _, _, _, = etcd._get_server(ETCD_POD_RANK, "0")

        return value

    @staticmethod
    def get_cluster(timeout=15):
        begin = time.time()
        etcd, lock = get_global_etcd()
        while True:
            if time.time() - begin > timeout:
                logger.warning("get pod leader error!")
                raise EdlBarrierError("get cluster error")

            leader_id = EtcdDB.get_pod_leader_id()
            if leader_id is None:
                time.sleep(1)
                continue

            with lock:
                try:
                    value, _, _, _, _, = etcd._get_server(ETCD_CLUSTER,
                                                          leader_id)
                except Exception as e:
                    logger.debug("get cluster of leader_id:{} error:{}".format(
                        leader_id, e))
                    time.sleep(1)
                    continue

            if value is None:
                time.sleep(1)
                continue

            cluster = Cluster()
            cluster.from_json(value)
            if len(cluster.pods) == 0:
                raise EdlBarrierError("get cluster error")

            return cluster

    @staticmethod
    def get_pod_leader(timeout=15):
        cluster = EtcdDB.get_cluster(timeout)
        return cluster.pods[0]

    @staticmethod
    def get_data_reader_leader():
        raise NotImplementedError()

    @staticmethod
    def get_diff_pods(cluster):
        all_inited, all_running, all_succeed, all_failed = EtcdDB.get_pods_status(
        )

        resource = EtcdDB.get_resource_pods_ids_set()
        last = cluster.get_pods_ids_set()

        added = now - last
        #diff = init.symmetric_difference(now)
        succeed = last & all_succeed
        failed = last - now - succeeded
        inited = now & all_inited

        return (inited, added, succeed, failed)

    @staticmethod
    def set_pod_flag(pod_id, flag):
        if not flag:
            EtcdDB.set_pod_status(pod.get_id(), JobStatus.ERROR)
            logger.fatal("local trainers meets error!")
            return

        EtcdDB.set_pod_status(pod.get_id(), JobStatus.COMPLETE)
        logger.info("local trainers succeeded!")

    @staticmethod
    def get_train_status():
        etcd, lock = get_global_etcd()

        leader_id = Etcd.get_pod_leader_id()
        with lock:
            value, _, _, _, _, = etcd._get_server(ETCD_TRAIN_STATUS, leader_id)

        if value is None:
            return None

        d = json.load(value)
        return d["status"]

    @staticmethod
    def set_train_status(pod_id, status):
        etcd, lock = get_global_etcd()
        service = ETCD_TRAIN_STATUS
        server = pod_id
        info = json.dumps({"status": int(status)})
        with lock:
            etcd.set_server_permanent(service, server, info)

    @staticmethod
    def wait_leader_exit():
        logger.info("begin to wait leader exit!")
        logger.info("leader exit so this pod exit!")

    @staticmethod
    def wait_resource_flag(cluster):
        logger.info("begin to wait resource exit!")
        logger.info("leader exit!")
