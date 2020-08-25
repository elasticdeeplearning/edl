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
from .cluster import Cluster
from .pod import Pod, JobStatus

from .global_vars import get_global_etcd, ETCD_POD_RANK, ETCD_POD_RESOURCE

import six


class Watcher(object):
    def __init__(self, job_env, cluster, pod):
        self._etcd = None

        self._job_id = job_env.job_id
        self._current_pod = pod
        self._cluster = copy.copy(cluster)
        logger.info("watcher gets the init cluster:{}".format(self._cluster))
        self._new_cluster = copy.copy(self._cluster)
        self._changed = False

        self._new_cluster = Cluster()
        self._lock = Lock()
        self._stop = Event()

        self._t_watcher = None

    def start(self):
        self._etcd = EtcdClient(etcd_endpoints, root=job_id)
        self._etcd.init()

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
                    self._cluster.from_rank_dict(ranks)
                    self._new_cluster = copy.copy(self._cluster)
                    continue

            with self._lock:
                self._new_cluster.from_rank_dict(ranks)

            if self._is_world_changed():
                break

            with self._lock:
                # update the cluster info.
                self._cluster = copy.copy(self._new_cluster)

            time.sleep(2)

    @property
    def changed(self):
        with self._lock:
            return self._changed

    def _is_world_changed(self):
        """
        list[Rank ordered pod_id] changed
        """

        with self._lock:
            old = self._cluster.get_pods_ids_list()
            new = self._new_cluster.get_pods_ids_list()

        if old != new:
            logger.debug("_is_world_changed find changed")
            with self._lock:
                self._changed = True
                old.changed_pods(new)
            return True

        return False

    def get_cluster(self):
        with self._lock:
            return self._cluster

    def get_new_cluster(self):
        with self._lock:
            return self._new_cluster

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_watcher:
                self._t_watcher.join()
                self._t_watcher = None
        logger.debug("watcher stopped")

    def is_stopped(self):
        with self._lock:
            return self._t_watcher == None

    def __exit__(self):
        self.stop()
