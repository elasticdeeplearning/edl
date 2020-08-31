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
    SUCCEED = 3
    FAILED = 4


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
    def get_pods_status(etcd=None, lock=None):
        if etcd is None:
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
            if status == int(JobStatus.FAILED):
                failed.add(server.server)
            elif status == int(JobStatus.SUCCEED):
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
            EtcdDB.set_job_status(pod.get_id(), JobStatus.SUCCEED)
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
    def get_resource_pods_dict(etcd=None, lock=None):
        if etcd is None:
            etcd, lock = get_global_etcd()

        with lock:
            servers = etcd.get_service(ETCD_POD_RESOURCE)

        pods = {}
        for s in servers:
            p = Pod()
            p.from_json(s.info)
            pods[p.get_id()] = p

        return pods

    @staticmethod
    def get_pod_leader_id():
        etcd, lock = get_global_etcd()
        with lock:
            value = etcd.get_value(ETCD_POD_RANK, "0")

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
                    value = etcd.get_value(ETCD_CLUSTER, leader_id)
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
    def get_current_cluster(etcd=None, lock=None):
        if etcd is None:
            etcd, lock = get_global_etcd()

        with lock:
            value = etcd.get_value(ETCD_CLUSTER, ETCD_CLUSTER)

        if value is None:
            return None

        cluster = Cluster()
        cluster.loads(value)
        return cluster

    @staticmethod
    def _generate_cluster_from_resource(etcd=None, lock=None, resource_pods):
        if etcd is None:
            etcd, lock = get_global_etcd()

        leader_id = Etcd.get_pod_leader_id(etcd, lock)
        if leader_id is None or len(resource_pods) <= 0:
            return None

        new_cluster = Cluster()
        pods = new_cluster.get_pods()
        if leader_id not in resource_pods:
            return None

        rank = 0
        pods.add(resource_pods[leader_id])
        # set rank
        pods[0].rank = rank
        rank += 1
        resource_pods.pop(leader_id)
        for pod_id, pod in six.iteritems(resource_pods):
            pod.rank = rank
            pods.append(pod)
            rank += 1

        new_cluster.new_stage()
        return new_cluster

    @staticmethod
    def _append_inited_pods(current_cluster, resource_pods, new_cluster):
        rank = current_cluster.get_pods_nranks()
        new_cluster = copy.copy(current_cluster)
        new_pods = new_cluster.get_pods()
        for pod_id, pod in six.iteritems(resource_pods):
            if pod.status == JobStatus.INITIAL:
                pod.rank = rank
                rank += 1
                new_pods.append(pod)

        if new_cluster.get_pods_nranks() != current_cluster.get_pods_nranks():
            new_cluster.new_stage()

    @staticmethod
    def _generate_cluster(etcd=None, lock=None):
        if etcd is None:
            etcd, lock = get_global_etcd()

        current_cluster = EtcdDB.get_current_cluster(etcd, lock)
        resource_pods = EtcdDB.get_resource_pods_dict(etcd, lock)

        if current_cluster is None:
            new_cluster = EtcdDB._generate_cluster_from_resource(etcd, lock,
                                                                 resource_pods)
            return None, new_cluster

        current_ids = current_cluster.get_pods_ids_set()
        resource_ids = resource_pods.keys()
        all_inited, all_running, all_succeed, all_failed = EtcdDB.get_pods_status(
            etcd, lock)

        disappeared = current_ids - resource_ids - all_inited - all_running - all_succeed - all_failed
        failed = current_ids & all_failed
        if len(disappeared) > 0 or len(failed) > 0:
            logger.warning("find disappeard pods:{} failed_pods:{}".format(
                disappeared, failed))
            return current_cluster, EtcdDB._generate_cluster_from_resource(
                etcd, lock, resource_pods)

        succeed = current_ids & all_succeed
        if len(succeed) > 0:
            logger.debug("find succeed pods:{}".format(succeed))
            new_cluster = copy.copy(current_cluster)
            return new_cluster

        running = current_ids & all_running
        inited = current_ids & all_inited
        if len(inited) > 0:
            train_status = EtcdDB.get_train_status(etcd, lock)
            if train_status == TrainStatus.INITIAL or train_status == TrainStatus.RUNNING:
                logger.info("find running pods:{} and init pods{}".format(
                    inited, running))
                EtcdDB._append_inited_pods(current_cluster, resource_pods,
                                           new_cluster)
                return current_cluster, new_cluster

        logger.debug("find succeed pods:{}".format(succeed))
        new_cluster = copy.copy(current_cluster)
        return current_cluster, new_cluster

    @staticmethod
    def _set_cluster_if_leader(cluster, pod, etcd_client, lock):
        leader_key = etcd_client.get_full_path(ETCD_POD_RANK, "0")
        cluster_key = etcd_client.get_full_path(ETCD_CLUSTER, ETCD_CLUSTER)

        with lock:
            etcd = etcd_client._etcd
            status, _ = etcd.transaction(
                compare=[
                    etcd.transactions.value(leader_key) == pod.get_id(),
                ],
                success=[
                    etcd.transactions.put(cluster_key, cluster.to_json()),
                ],
                failure=[])

        return status

    @staticmethod
    def generate_cluster(job_env):
        if etcd is None:
            etcd, lock = get_global_etcd()

        current_cluster, new_cluster = EtcdDB._generate_cluster(etcd, lock,
                                                                job_env)
        if new_cluster is None:
            logger.warning("can't generate new cluster")
            return False

        if new_cluster.get_pods_nranks() < job_env.min_nodes:
            new_cluster.status = JobStatus.FAILED
        elif new_cluster.get_pods_nranks() > job_env.max_nodes:
            pods = new_cluster.get_pods()
            pods = pods[0:job_env.max_nodes]

        if current_cluster is None or current_cluster.stage != new_cluter.stage:
            logger.info("generate new cluster:{}".format(new_cluster))
            return etcd._set_cluster_if_leader(new_cluster, pod, etcd, lock)

        return True

    @staticmethod
    def set_pod_flag(pod_id, flag):
        if not flag:
            EtcdDB.set_pod_status(pod.get_id(), JobStatus.FAILED)
            logger.fatal("local trainers meets error!")
            return

        EtcdDB.set_pod_status(pod.get_id(), JobStatus.SUCCEED)
        logger.info("local trainers succeeded!")

    @staticmethod
    def get_train_status(etcd=None, etcd_lock=None):
        if etcd is None:
            etcd, lock = get_global_etcd()

        leader_id = Etcd.get_pod_leader_id()
        with lock:
            value = etcd.get_value(ETCD_TRAIN_STATUS, leader_id)

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
    def wait_resource(pod, timeout=15):
        pods = EtdbDB.get_resource_pods_dict()
        if len(pods) == 1:
            if pod.get_id() in pods:
                return True

        if len(pods) == 0:
            return True

        return False
