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
import copy
from .cluster import Cluster, Pod, JobStatus

from .global_vars import get_global_etcd, ETCD_POD_RANK, ETCD_POD_RESOURCE

import six


class Watcher(object):
    def __init__(self, etcd_endpoints, job_id, current_pod):
        self._etcd = EtcdClient(etcd_endpoints, root=job_id)
        self._etcd.init()

        self._job_id = job_id
        self._current_pod = current_pod

        # servers in etcd
        #self._ranks = None  # {rank:pod_json}

        self._cluster = Cluster()
        self._new_cluster = Cluster()
        self._lock = Lock()
        self._stop = Event()

        # running pods
        servers = self._etcd.get_service(ETCD_POD_RANK)
        #self._ranks = {}
        ranks = {}
        for s in servers:
            ranks[int(s.server)] = s.info
        self._cluster.from_json(ranks)
        self._new_cluster = copy.copy(self._cluster)
        self._changed = False
        logger.info("watcher gets the init cluster:{}".format(self._cluster))

        self._t_watcher = Thread(target=self._watcher)
        self._t_watcher.start()

    def _watcher(self):
        begin = time.time()
        while not self._stop.is_set():
            servers = self._etcd.get_service(ETCD_POD_RANK)
            ranks = {}
            for s in servers:
                ranks[int(s.server)] = s.info

            with self._lock:
                if ranks is None:
                    #self._ranks = ranks
                    self._cluster.from_json(ranks)
                    self._new_cluster = copy.copy(self._cluster)
                    continue

            self._clear_changed()

            with self._lock:
                self._new_cluster.from_json(ranks)

            if self._is_world_changed(self._new_cluster):
                break

            with self._lock:
                # update the cluster info.
                self._cluster = copy.copy(self._new_cluster)

            time.sleep(2)

    def changed(self):
        with self._lock:
            return self._changed

    def _is_world_changed(self):
        """
        list[Rank ordered pod_id] changed
        """

        old = self._cluster.get_pods_ids()
        with self._lock:
            new = self._new_cluster.get_pods_ids()

        if old != new:
            with self._lock:
                self._changed = True
            logger.info("cluster change from pods:{} to pods:{}".format(old,
                                                                        new))
            return True

        return False

    def get_cluster(self):
        with self._lock:
            return self._cluster, self._new_cluster

    def stop(self):
        self._stop.set()
        if self._t_watcher:
            self._t_watcher.join()
            with self._lock:
                self._t_watcher = None
        logger.debug("watcher stopped")

    def __exit__(self):
        self.stop()


def get_current_pod_ids_from_resource():
    etcd, lock = get_global_etcd()
    with lock:
        pod_resource_servers = etcd.get_service(ETCD_POD_RESOURCE)

    p = Pod()
    ids = set()
    for m in pod_resource_servers:
        p.from_json(m.info)
        ids.add(p.get_id())

    return ids


def get_cluster():
    etcd, lock = get_global_etcd()
    cluster = Cluster()
    servers = etcd.get_service(ETCD_POD_RANK)
    ranks = {}
    for s in servers:
        ranks[int(s.server)] = s.info
    cluster.from_json(ranks)
    return cluster


def get_pod_leader():
    etcd, lock = get_global_etcd()
    with lock:
        value, _, _, _, _, = etcd._get_server(ETCD_POD_RANK, "0")

    leader = Pod()
    leader.from_json(value)
    return leader


def get_data_reader_leader():
    #raise NotImplementedError()
    pass
