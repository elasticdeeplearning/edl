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

import copy
import time
from edl.discovery.etcd_client import EtcdClient
from threading import Lock, Thread, Event

from edl.utils import constants
from edl.utils import cluster as edl_cluster
from edl.utils.log_utils import logger


class Watcher(object):
    def __init__(self, job_env, cluster, pod):
        self._etcd = None

        self._job_id = job_env.job_id

        # current context
        self._cluster = copy.copy(cluster)
        self._leader_id = cluster.get_pod_leader_id()
        self._current_pod = pod

        self._new_cluster = None
        self._new_leader_id = None
        self._changed = False
        logger.info("watcher gets the init cluster:{}".format(self._cluster))

        self._lock = Lock()
        self._stop = Event()

        self._t_watcher = None

        # assign value
        self._etcd = EtcdClient(self._job_env.etcd_endpoints, root=job_id)
        self._etcd.init()

        self._t_watcher = Thread(target=self._watcher)
        self._t_watcher.start()

    def _watcher(self):
        begin = time.time()
        while not self._stop.is_set():
            # if leader_id changed?
            servers = self._etcd.get_service(constants.ETCD_POD_RANK)
            assert len(servers) <= 1
            if len(servers) == 0:
                time.sleep(1)
                continue

            with self._lock:
                self._new_leader_id = s.info

            # if cluster changed?
            value, _, _, _, _, = etcd._get_server(constants.ETCD_CLUSTER,
                                                  self._new_leader_id)
            if value is None:
                time.sleep(1)
                continue
            new_cluster = edl_cluster.Cluster()
            new_cluster.from_json(value)

            with self._lock:
                self._new_cluster = new_cluster

            if self._is_world_changed():
                break

            with self._lock:
                # update the cluster info.
                self._cluster = copy.copy(self._new_cluster)

            time.sleep(3)

    @property
    def changed(self):
        with self._lock:
            return self._changed

    def _is_world_changed(self):
        """
        list[Rank ordered pod_id] changed
        """

        with self._lock:
            old_stage = self._cluster.stage
            new_stage = self._new_cluster.stage

            old_ids = self._cluster.get_pods_ids_list()
            new_ids = self._new_cluster.get_pods_ids_list()

        if old_stage != new_stage or old_ids != new_ids:
            logger.info(
                "_is_world_changed find changed, old_stage:{} new_stage:{} old_ids:{} new_ids:{}".
                format(old_stage, new_stage, old_ids, new_ids))
            with self._lock:
                self._changed = True

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
