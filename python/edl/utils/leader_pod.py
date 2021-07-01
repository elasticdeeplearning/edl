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
from edl.utils import constants
from edl.utils import error_utils
from edl.utils import etcd_utils
from edl.utils import exceptions
from edl.utils import string_utils

from edl.utils.log_utils import logger
from edl.discovery import etcd_client
from edl.utils import resource_pods


class Register(object):
    def __init__(self, job_env, pod_id, cluster_generator, ttl=constants.ETCD_TTL):
        self._job_env = job_env
        self._is_leader = False
        self._pod_id = pod_id
        self._ttl = ttl
        self._generate_cluster = cluster_generator

        self._stop = threading.Event()
        self._service_name = constants.ETCD_POD_RANK
        self._server = "0"
        self._lock = threading.Lock()

        self._etcd = None
        self._t_register = None

        # assign value
        self._etcd = etcd_client.EtcdClient(
            self._job_env.etcd_endpoints,
            root=self._job_env.job_id,
            timeout=constants.ETCD_CONN_TIMEOUT,
        )
        self._etcd.init()

        self._seize_leader()

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _seize_leader(self, timeout=constants.ETCD_CONN_TIMEOUT):
        info = self._pod_id

        if not self._etcd.set_server_not_exists(
            self._service_name,
            self._server,
            info=info,
            timeout=constants.ETCD_CONN_TIMEOUT,
            ttl=self._ttl,
        ):
            logger.debug(
                "Can't seize leader on etcd key:{}".format(
                    self._etcd.get_full_path(self._service_name, self._server)
                )
            )

            with self._lock:
                self._is_leader = False

            self._generate_cluster.stop()
            return False

        with self._lock:
            self._is_leader = True

        self._generate_cluster.start()
        logger.info(
            "register leader:{} on etcd key:{}".format(
                info, self._etcd.get_full_path(self._service_name, self._server)
            )
        )
        return True

    def is_leader(self):
        with self._lock:
            return self._is_leader

    def _refresh(self):
        try:
            self._etcd.refresh(self._service_name, self._server, ttl=constants.ETCD_TTL)
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
                logger.fatal(str(e))
                break

            time.sleep(3)

    def stop(self):
        self._stop.set()
        if self._t_register:
            self._t_register.join()

            with self._lock:
                self._t_register = None

            self._etcd.remove_server(self._service_name, self._server)
            self._generate_cluster.stop()
            logger.info("pod:{} leader_register stopped".format(self._pod_id))

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def is_stopped(self):
        with self._lock:
            return self._t_register is None


@error_utils.handle_errors_until_timeout
def get_pod_leader_id(etcd, timeout=15):
    value = etcd.get_value(constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER)
    if value is None:
        return None

    return string_utils.bytes_to_string(value)


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, timeout=15):
    leader_id = get_pod_leader_id(etcd, timeout=timeout)

    if leader_id is None:
        raise exceptions.EdlTableError(
            "leader_id={}:{}".format(etcd_utils.get_rank_table_key(), leader_id)
        )

    pods = resource_pods.load_from_etcd(etcd, timeout=timeout)
    if leader_id not in pods:
        raise exceptions.EdlTableError(
            "leader_id:{} not in resource pods".format(leader_id)
        )

    return pods[leader_id]
