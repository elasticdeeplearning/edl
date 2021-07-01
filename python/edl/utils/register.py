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
from edl.discovery import etcd_client
from edl.utils import constants
from edl.utils.log_utils import logger


class Register(object):
    def __init__(
        self, etcd_endpoints, job_id, service, server, info, ttl=constants.ETCD_TTL
    ):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        self._etcd = None
        self._t_register = None
        self._lock = threading.Lock()
        self._info = info
        self._ttl = ttl

        self._etcd = etcd_client.EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=ttl
        )
        self._etcd.init()

        try:
            self._etcd.set_server_not_exists(service, server, self._info, ttl=ttl)
            logger.info(
                "register pod:{} in etcd path:{}".format(
                    info, self._etcd.get_full_path(service, server)
                )
            )
        except Exception as e:
            logger.fatal(
                "connect to etcd:{} error:{} service:{} server:{} info:{}".format(
                    etcd_endpoints, e, service, server, info
                )
            )
            raise e

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _refresher(self):
        while not self._stop.is_set():
            try:
                self._etcd.refresh(self._service, self._server)
                time.sleep(self._ttl / 2)
            except Exception as e:
                logger.fatal(
                    "register meet error and exit! class:{} error:{}".format(
                        self.__class__.__name__, e
                    )
                )
                # if refresher stopped, the pod will exit from the cluster
                break

    def stop(self):
        self._stop.set()
        if self._t_register:
            self._t_register.join()

            with self._lock:
                self._t_register = None

            self._etcd.remove_server(self._service, self._server)

    def is_stopped(self):
        with self._lock:
            return self._t_register is None

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
