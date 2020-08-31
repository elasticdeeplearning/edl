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
from .cluster import Cluster


class GenerateCluster(object):
    def __init__(self):
        self._cluster = Cluster()
        self._service = service
        self._server = server
        self._stop = threading.Event()
        self._etcd = None
        self._t_register = None
        self._lock = threading.Lock()

    def start(self):
        self._etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=6)
        self._etcd.init()

        self._generate_cluster()

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _generate_cluster(timeout=600):
        begin = time.time()
        while True:
            try:
                if EtcdDB.generate_cluster(self._etcd, self._lock):
                    return True

                raise EdlGenerateClusterError("can't generate cluster")
            except Exception as e:
                if time.time() - begin >= timeout:
                    logger.fatal("can't generate cluster exit!")

                time.sleep(3)
                raise e

    def _refresher(self):
        while not self._stop.is_set():
            try:
                self._generate_cluster(timeout=60)
            except Exception as e:
                break

            time.sleep(3)

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_register:
                self._t_register.join()
                self._t_register = None

        logger.info("{} exit".format(self.__class__.__name__))

    def is_stopped(self):
        with self._lock:
            return self._t_register == None

    def __exit__(self):
        self.stop()

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
