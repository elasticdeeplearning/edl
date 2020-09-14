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

from . import cluster
from .log_utils import logger
from ..discovery.etcd_client import EtcdClient
from . import constants


class LeaderRegister(object):
    def __init__(self, job_env, pod_id):
        self._job_env = job_env
        self._is_leader = False
        self._pod_id = pod_id
        self._generate_cluster = cluster.ClusterGenerator(job_env, pod_id)

        self._stop = threading.Event()
        self._service_name = constants.ETCD_POD_RANK
        self._server = "0"
        self._lock = threading.Lock()

        self._etcd = None
        self._t_register = None

        # assign value
        self._etcd = EtcdClient(
            self._job_env.etcd_endpoints,
            root=self._job_env.job_id,
            timeout=constants.ETCD_CONN_TIMEOUT)
        self._etcd.init()

        self._seize_leader()

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _seize_leader(self, timeout=constants.ETCD_CONN_TIMEOUT):
        begin = time.time()
        info = self._pod_id

        if not self._etcd.set_server_not_exists(
                self._service_name,
                self._server,
                info=info,
                timeout=constants.ETCD_CONN_TIMEOUT,
                ttl=constants.ETCD_TTL):
            logger.debug("Can't seize leader on etcd key:{}".format(
                self._etcd.get_full_path(self._service_name, self._server)))

            with self._lock:
                self._is_leader = False

            self._generate_cluster.stop()
            return False

        with self._lock:
            self._is_leader = True

        self._generate_cluster.start()
        logger.info("register leader:{} on etcd key:{}".format(
            info, self._etcd.get_full_path(self._service_name, self._server)))
        return True

    def is_leader(self):
        with self._lock:
            return self._is_leader

    def _refresh(self):
        try:
            self._etcd.refresh(
                self._service_name, self._server, ttl=constants.ETCD_TTL)
            return True
        except Exception as e:
            logger.warning("refresh error:{}".format(e))
            with self._lock:
                self._is_leader = False
        return False

    def _refresher(self):
        while not self._stop.is_set():
            with self._lock:
                is_leader = self._is_leader

            try:
                if is_leader:
                    self._refresh()
                else:
                    self._seize_leader()
            except Exception as e:
                # exit when error ocurred
                break

            time.sleep(3)

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_register:
                self._t_register.join()
                self._t_register = None
                self._etcd.remove_server(self._service_name, self._server)

                self._generate_cluster.stop()

        logger.info("pod_register stopped")

    def __exit__(self):
        self.stop()

    def is_stopped(self):
        with self._lock:
            return self._t_register == None
