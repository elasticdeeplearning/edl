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

from threading import Lock, Thread, Event
import time
from utils import logger
from edl.discovery.etcd_client import EtcdClient
import json
import collections
from .cluster import Cluster, Pod

from .global_vars import get_etcd, ETCD_POD_RANK, ETCD_POD_RESOURCE

import six


class Watcher(object):
    def __init__(self, etcd_endpoints, job_id, current_pod):
        self._etcd = EtcdClient(etcd_endpoints, root=job_id)
        self._etcd.init()

        self._job_id = job_id
        self._changed = False
        self._current_pod = current_pod

        # servers in etcd
        self._ranks = None  # {rank:pod_json}

        self._cluster = Cluster()
        self._lock = Lock()
        self._stop = Event()

        self._leader_changed = False
        self._failed_pods = []
        self._changed_follower_pods = []

        servers = self._etcd.get_service(ETCD_POD_RANK)
        ranks = {}
        for s in servers:
            ranks[int(s.server)] = s.info
        self._cluster.from_json(ranks)
        logger.info("watcher gets the init cluster:{}", self._cluster)

        self._t_watcher = Thread(target=self._watcher)
        self._t_watcher.start()

    def _watcher(self):
        begin = time.time()
        while not self._stop.is_set():
            servers = self._etcd.get_service(ETCD_POD_RANK)
            ranks = {}
            for s in servers:
                ranks[int(s.server)] = s.info

            new_cluster = Cluster()
            with self._lock:
                if self._ranks is None:
                    self._ranks = ranks
                    self._cluster.from_json(ranks)
                    continue

                if not self._is_cluster_changed(self._ranks, ranks):
                    time.sleep(1)
                    continue

                self._changed = True

            self._clear_changed()

            new_cluster.from_json(ranks)
            if self._is_any_pod_failed(new_cluster):
                break

            if self._is_leader_changed(new_cluster):
                # leader will not find self changed.
                break

            if self._is_follower_changed(new_cluster):
                break

    def _clear_changed(self):
        with self._lock:
            self._job_stage_changed = False
            self._failed_pods = []
            self._changed_follower_pods = []

    def _is_leader_changed(self, new_cluster):
        """
        1. leader pod id changed
        2. leader stage changed 
        """
        if len(self._cluster.pods) == 0:
            assert False, "internal error, can't reach here"
            return False

        if len(new_cluster.pods) == 0:
            with self._lock:
                self._leader_changed = True
            return True

        new_leader = new_cluster.pods[0]
        old_leader = self._cluster.pods[0]

        if new_leader.get_id() != old_leader.get_id() or \
                new_leader.stage != old_leader.stage():
            with self._lock:
                self._leader_changed = True
            return True

        return False

    def is_leader_changed(new_cluster):
        with self._lock:
            return self._leader_changed

    def _is_follower_changed(self, old, new):
        """
        1. some pod disappear
        2. some pod add to cluster
        3. rank of pod changed
        """
        if len(self._cluster.pods) == 0:
            assert False, "internal error, can't reach here"
            return False

        if len(new_cluster.pods) == 0:
            self._changed_follower_pods = self._cluster.pods
            return True

        filer_ids = set()
        # find the changed pods
        for old_pod in self._cluster.pods:
            new_pod = new_cluster.get_pod_by_id(old_pod.get_id())

            if old_pod.rank != new_pod.rank:
                with self._lock:
                    self._changed_follower_pods.append(old_pod)
            filer_ids.add(old_pod.get_id())

        # find the new added pods
        for pod in new_cluster.pods:
            if pod.get_id not in filer_ids:
                with self._lock:
                    self._changed_follower_pods.append(pod)

        with self._lock:
            return len(self._changed_follower_pods) != 0

    def is_follower_changed(self):
        with self._lock:
            return self._changed_follower_pods != None

    def _is_any_pod_failed(self, new_cluster):
        for pod in new_cluster.pods:
            if pod.status == PodStatus.ERROR:
                with self._lock:
                    self._failed_pods.append(pod)

        with self._lock:
            return len(self._failed_pods) != 0

    def get_failed_pods(self):
        with self._lock:
            return self._failed_pods

    def _is_cluster_changed(self, old, new):
        for k, v in six.iteritems(old):
            if k not in new:
                logger.info(
                    "train world changed, old_cluster k:{} not in new_cluster:{}".
                    format(k, new))
                return True

            if old[k] != new[k]:
                logger.info(
                    "train world changed, old_cluster k:{}=>v:{} != new_cluster k:{}=>v:{}".
                    format(k, old[k], k, new[k]))
                return True

        return False

    def get_cluster(self):
        with self._lock:
            return self._cluster

    def stop(self):
        self._stop.set()
        self._t_watcher.join()

    def __exit__(self):
        self.stop()


def get_current_pod_ids_from_resource():
    etcd, lock = get_etcd()
    with lock:
        pod_resource_servers = etcd.get_service(ETCD_POD_RESOURCE)

    p = Pod()
    ids = set()
    for m in pod_resource_servers:
        p.from_json(m.info)
        ids.add(p.get_id())

    return ids


def get_cluster():
    etcd, lock = get_etcd()
    cluster = Cluster()
    servers = etcd.get_service(ETCD_POD_RANK)
    ranks = {}
    for s in servers:
        ranks[int(s.server)] = s.info
    cluster.from_json(ranks)
    return cluster


def get_pod_leader():
    etcd, lock = get_etcd()
    with lock:
        value, _, _, _, _, = etcd._get_server(ETCD_POD_RANK, "0")

    leader = Pod()
    leader.from_json(value)
    return leader


def get_data_reader_leader():
    #raise NotImplementedError()
    pass
