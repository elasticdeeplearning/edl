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

from .etcd_db import get_global_etcd
from .log_utils import logger
from ..discovery.etcd_client import EtcdClient
from . import constants


class Register(object):
    def __init__(self, etcd_endpoints, job_id, service, server, info):
        self._service = service
        self._server = server
        self._stop = threading.Event()
        self._etcd = None
        self._t_register = None
        self._lock = threading.Lock()
        self._info = info

        self._etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=6)
        self._etcd.init()

        try:
            self._etcd.set_server_not_exists(
                service, server, self._info, ttl=constants.ETCD_TTL)
            logger.info("register pod:{} in etcd path:{}".format(
                info, self._etcd.get_full_path(service, server)))
        except Exception as e:
            logger.fatal(
                "connect to etcd:{} error:{} service:{} server:{} info:{}".
                format(etcd_endpoints, e, service, server, info))
            raise e

        self._t_register = threading.Thread(target=self._refresher)
        self._t_register.start()

    def _refresher(self):
        while not self._stop.is_set():
            try:
                self._etcd.refresh(self._service, self._server)
                time.sleep(3)
            except Exception as e:
                logger.fatal("register meet error and exit! class:{} error:{}".
                             format(self.__class__.__name__, e))
                break

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._t_register:
                self._t_register.join()
                self._t_register = None

                self._etcd.remove_server(self._service, self._server)

        logger.info("{} exit".format(self.__class__.__name__))

    def is_stopped(self):
        with self._lock:
            return self._t_register == None

    def __exit__(self):
        self.stop()


class PodResourceRegister(Register):
    def __init__(self, job_env, pod):
        service = constants.ETCD_POD_RESOURCE
        server = "{}".format(pod.get_id())
        value = pod.to_json()

        super(PodResourceRegister, self).__init__(
            etcd_endpoints=job_env.etcd_endpoints,
            job_id=job_env.job_id,
            service=service,
            server=server,
            info=value)

        db = get_global_etcd()
        db.get_resource_pods_dict()


class DataReaderRegister(Register):
    def __init__(self, etcd_endoints, job_id, pod_id, reader):
        service = constants.ETCD_POD_DATA_READER
        sever = pod_id
        value = {
            "id": reader.get_id(),
            "name": reader.name,
            "endpoint": reader.endpoint,
        }

        info = value.dumps(value)

        super(DataReaderRegister, self).__init__(
            etcd_endpoints=etcd_endpoints,
            job_id=job_id,
            service=service,
            server=server,
            info=value)
