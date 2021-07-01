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
import threading
import time
from edl.discovery import etcd_client
from edl.utils import cluster as edl_cluster
from edl.utils.log_utils import logger


class Watcher(object):
    def __init__(self, job_env, cluster):
        self._job_id = job_env.job_id

        # current context
        self._cluster = copy.copy(cluster)

        self._new_cluster = cluster
        self._changed = False
        logger.info("watcher gets the init cluster:{}".format(self._cluster))

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._etcd = None
        self._t_watcher = None
        self._job_env = job_env

        # assign value
        self._etcd = etcd_client.EtcdClient(
            self._job_env.etcd_endpoints, root=self._job_id
        )
        self._etcd.init()

        self._t_watcher = threading.Thread(target=self._watcher)
        self._t_watcher.start()

    def _watcher(self):
        while not self._stop.is_set():
            # if cluster changed?
            new_cluster = edl_cluster.wait_to_load_from_etcd(self._etcd, timeout=60)
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
                "_is_world_changed find changed, \
                old_stage:{} new_stage:{} old_ids:{} new_ids:{}".format(
                    old_stage, new_stage, old_ids, new_ids
                )
            )
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
        if self._t_watcher:
            self._t_watcher.join()

            with self._lock:
                self._t_watcher = None

            logger.info("watcher stopped")

    def is_stopped(self):
        with self._lock:
            return self._t_watcher is None

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
