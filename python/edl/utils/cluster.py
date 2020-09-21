# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import copy
import json
import six
import threading
import time
import traceback
import uuid
from edl.discovery import etcd_client
from edl.utils import cluster as edl_cluster
from edl.utils import constants
from edl.utils import error_utils
from edl.utils import exceptions
from edl.utils import json_serializable
from edl.utils import leader_pod
from edl.utils import pod as edl_pod
from edl.utils import resource_pods as edl_resource_pods
from edl.utils import status as edl_status
from edl.utils import train_status as edl_train_status
from edl.utils.log_utils import logger


class Cluster(json_serializable.Serializable):
    def __init__(self):
        self._pods = []
        self._stage = None
        self._status = edl_status.Status.INITIAL

    def __str__(self):
        return "pods:{} job_stage:{} status:{}".format(
            [str(pod) for pod in self._pods], self._stage, self._status)

    def details(self):
        return "pods:{} job_stage:{} status:{}".format(
            [pod.details() for pod in self._pods], self._stage, self._status)

    def update_pods(cluster):
        self._pods = copy.copy(cluster.pods)

    def get_trainers_nranks(self):
        return len(self.get_trainers_endpoints())

    def get_pods_nranks(self):
        return len(self._pods)

    def get_pods(self):
        return self._pods

    @property
    def pods(self):
        return self._pods

    def get_trainers_endpoints(self):
        r = []
        for pod in self._pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def get_pods_endpoints(self):
        r = []
        for pod in self._pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self._pods:
            if str(pod_id) == str(pod._id):
                return pod

        return None

    def get_pods_ids_list(self):
        ids = []
        for pod in self._pods:
            ids.append(pod._id)

        return ids

    def get_pods_ids_set(self):
        ids = set()
        for pod in self._pods:
            ids.add(pod._id)

        return ids

    # FIXME(gongwb): use from_pb, to_pb later
    def to_json(self):
        d = {}
        for i, pod in enumerate(self._pods):
            d[i] = pod.to_json()

        d = {"pods": d, "stage": self._stage, "status": int(self._status)}

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)
        pods = d["pods"]
        self._stage = d["stage"]
        self._status = d["status"]

        od = collections.OrderedDict(sorted(d["pods"].items()))
        pods = []
        for i, (key, value) in enumerate(six.iteritems(od)):
            pod = edl_pod.Pod()
            if i != int(key):
                raise exceptions.EdlRankError(
                    "rank:{} is not exists in {}".format(i, d))
            pod.from_json(value)
            pods.append(pod)

        self._pods = pods

    def get_leader_endpoint(self):
        assert len(self._pods) > 0
        return "{}:{}".format(self._pods[0].addr, self._pods[0].port)

    def get_leader_id(self):
        assert len(self._pods) > 0
        return self._pods[0].get_id()

    def new_stage(self):
        self._stage = str(uuid.uuid1())

    @property
    def stage(self):
        return self._stage

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        self._status = s


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, timeout=60):
    value = etcd.get_value(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER)

    if value is None:
        return None

    cluster = Cluster()
    cluster.from_json(value)
    return cluster


class Generator(object):
    def __init__(self, job_env, pod_id):
        self._cluster = edl_cluster.Cluster()
        self._service = constants.ETCD_CLUSTER
        self._server = constants.ETCD_CLUSTER
        self._stop = threading.Event()
        self._etcd = None
        self._t_register = None
        self._lock = threading.Lock()
        self._job_env = job_env
        self._pod_id = pod_id

    def start(self):
        self._etcd = etcd_client.EtcdClient(
            endpoints=self._job_env._etcd_endpoints,
            root=self._job_env.job_id,
            timeout=6)
        self._etcd.init()

        try:
            self._generate_cluster_and_check()
        except:
            pass

        self._t_register = threading.Thread(target=self._generate_cluster)
        self._t_register.start()

    def _generate_cluster(self, timeout=600):
        begin = time.time()
        while not self._stop.is_set():
            try:
                self._generate_cluster_and_check()

                begin = time.time()
                logger.debug("generate cluster ok!")
                time.sleep(3)
            except Exception as e:
                if time.time() - begin >= timeout:
                    raise e

                time.sleep(3)
                logger.debug("_generate_cluster error:{} {}".format(
                    e, traceback.format_exc()))

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_register:
                self._t_register.join()
                self._t_register = None

        logger.debug("{} exit".format(self.__class__.__name__))

    def is_stopped(self):
        with self._lock:
            return self._t_register == None

    def __exit__(self):
        self.stop()

    def _generate_cluster_from_resource(self, resource_pods):
        leader_id = leader_pod.get_pod_leader_id(self._etcd, timeout=15)
        if leader_id is None:
            raise exceptions.EdlTableError("leader key={}:{}".format(
                self._etcd.get_full_path(constants.ETCD_POD_RESOURCE,
                                         constants.ETCD_POD_RANK), leader_id))

        if leader_id not in resource_pods:
            raise exceptions.EdlTableError(
                "leader error, leader:{} not in resource:{}".format(
                    leader_id, resource_pods.keys()))

        new_cluster = edl_cluster.Cluster()
        pods = new_cluster.get_pods()

        rank = 0
        pods.append(resource_pods[leader_id])
        assert len(pods) == 1
        # set rank
        pods[0].rank = rank
        rank += 1

        resource_pods.pop(leader_id)
        for pod_id, pod in six.iteritems(resource_pods):
            if rank >= self._job_env.max_nodes:
                break

            pod.rank = rank
            pods.append(pod)
            rank += 1

        new_cluster.new_stage()
        return new_cluster

    def _append_inited_pods(self, current_cluster, resource_pods, new_cluster):
        rank = current_cluster.get_pods_nranks()
        new_cluster = copy.copy(current_cluster)
        new_pods = new_cluster.get_pods()

        ids = current_cluster.get_pods_ids_set()
        for pod_id, pod in six.iteritems(resource_pods):
            if pod.status == edl_status.Status.INITIAL \
                    and pod.get_pod_id() not in ids \
                    and len(new_pods) < self._job_env.max_nodes:
                pod.rank = rank
                rank += 1
                new_pods.append(pod)

        if new_cluster.get_pods_nranks() != current_cluster.get_pods_nranks():
            new_cluster.new_stage()

    def _generate_cluster_once(self):
        current_cluster = edl_cluster.load_from_etcd(self._etcd, timeout=15)
        resource_pods = edl_resource_pods.load_from_etcd(
            self._etcd, timeout=15)

        if len(resource_pods) <= 0:
            raise exceptions.EdlTableError("resource pods key={}:[]".format(
                self._etcd.get_full_path(constants.ETCD_POD_RESOURCE,
                                         self._pod_id)))

        if current_cluster is None:
            new_cluster = self._generate_cluster_from_resource(resource_pods)
            return None, new_cluster

        current_ids = current_cluster.get_pods_ids_set()
        resource_ids = set(resource_pods.keys())
        all_inited, all_running, all_succeed, all_failed = \
            edl_status.load_pods_status_from_etcd(self._etcd, timeout=15)

        disappeared = current_ids - resource_ids - all_inited - all_running - all_succeed - all_failed
        failed = current_ids & all_failed
        if len(disappeared) > 0 or len(failed) > 0:
            logger.warning("find disappeard pods:{} failed_pods:{}".format(
                disappeared, failed))
            return current_cluster, self._generate_cluster_from_resource(
                resource_pods)

        succeed = current_ids & all_succeed
        if len(succeed) > 0:
            logger.debug("find succeed pods:{}".format(succeed))
            new_cluster = copy.copy(current_cluster)
            return current_cluster, new_cluster

        running = current_ids & all_running
        inited = current_ids & all_inited
        if len(inited) > 0 and \
                current_cluster.get_pods_nranks() < self._job_env.max_nodes:
            train_status = edl_train_status.load_from_etcd(
                self._etcd, timeout=30)
            if train_status == edl_train_status.TrainStatus.INITIAL or train_status == edl_train_status.TrainStatus.RUNNING:
                logger.info("find running pods:{} and init pods{}".format(
                    inited, running))
                self._append_inited_pods(current_cluster, resource_pods,
                                         new_cluster)
                return current_cluster, new_cluster

        if len(succeed) > 0:
            logger.debug("find succeed pods:{}".format(succeed))

        new_cluster = copy.copy(current_cluster)
        return current_cluster, new_cluster

    @error_utils.handle_errors_until_timeout
    def _set_cluster_if_leader(self, cluster, timeout=120):
        leader_key = self._etcd.get_full_path(constants.ETCD_POD_RANK,
                                              constants.ETCD_POD_LEADER)
        cluster_key = self._etcd.get_full_path(constants.ETCD_CLUSTER,
                                               constants.ETCD_CLUSTER)

        etcd = self._etcd._etcd
        status, _ = etcd.transaction(
            compare=[etcd.transactions.value(leader_key) == self._pod_id, ],
            success=[etcd.transactions.put(cluster_key, cluster.to_json()), ],
            failure=[])

        message = "pod_id:{} leader_id:{} _set_cluster_if_leader status:{}".format(
            self._pod_id,
            leader_pod.get_pod_leader_id(
                self._etcd, timeout=15),
            status)

        if not status:
            raise exceptions.EdlEtcdIOError(message)

        return status

    def _generate_cluster_and_check(self):
        current_cluster, new_cluster = self._generate_cluster_once()

        if new_cluster.get_pods_nranks() < self._job_env.min_nodes:
            message = "new cluster pods size:{} ids:{} wait job_env range:[{}:{}]".format(
                new_cluster.get_pods_nranks(),
                new_cluster.get_pods_ids_set(), self._job_env.min_nodes,
                self._job_env.max_nodes)
            #new_cluster.status = Status.FAILED
            raise exceptions.EdlGenerateClusterError(message)

        if current_cluster is None or current_cluster.stage != new_cluster.stage:
            logger.info("current_cluster:{} to  new_cluster:{}".format(
                current_cluster, new_cluster))
            self._set_cluster_if_leader(new_cluster, timeout=120)
