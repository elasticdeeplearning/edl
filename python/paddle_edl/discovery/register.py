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

import logging
import socket
import time
import threading

from contextlib import closing
from .etcd_client import EtcdClient
from .server_alive import is_server_alive

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class ServerRegister(object):
    def __init__(self, db_endpoints, db_passwd=None, db_type='etcd'):
        self._db = EtcdClient(db_endpoints, db_passwd)
        self._server = None
        self._service_name = None
        self._is_db_connect = False

    def _monitor(self):
        # Todo, monitor cpu, gpu, net
        info = '{gpu:20%, net:1}'
        return info

    def _register(self, service_name, server, ttl=120):
        all_time = ttl
        while not is_server_alive(server)[0] and ttl > 0:
            logging.warning(
                'start to register, but server is not alive, ttl={}'.format(
                    ttl))
            ttl -= 2
            time.sleep(2)

        if ttl <= 0:
            logging.error('server is not up in time={}s'.format(all_time))
            raise Exception('server up timeout')

        self._db.set_server_not_exists(service_name, server, self._monitor())
        logging.info('register server={} success'.format(server))

    def _heartbeat(self, service_name, server, beat_time=1.5):
        retry = 45
        failed_count = 0

        while failed_count < retry:
            while is_server_alive(server)[0]:
                if failed_count != 0:
                    self._db.set_server_not_exists(service_name, server,
                                                   self._monitor())
                    failed_count = 0
                logging.debug(self._db._get_server(service_name, server))
                self._db.refresh(service_name, server)
                time.sleep(beat_time)
            failed_count += 1

            logging.warning('server={} is not alive, retry={}'.format(
                server, failed_count))
            time.sleep(2)

        logging.error('wait server restart timeout, exit')

    def register(self,
                 service_name,
                 server,
                 use_back_thread=False,
                 thread_daemon=False):
        """ register server forever if not use_back_thread, if use thread, thread will
            register server forever.
        """
        if self._is_db_connect is False:
            self._db.init()
            self._is_db_connect = True

        # register server forever
        if not use_back_thread:
            self._register(service_name, server)
            self._heartbeat(service_name, server)
        else:
            thread = threading.Thread(
                target=self.register, args=(service_name, server))
            thread.daemon = thread_daemon
            thread.start()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Server Register')
    parser.add_argument(
        '--db_endpoints',
        type=str,
        default='127.0.0.1:2379',
        help='database endpoints, e.g. 127.0.0.1:2379,127.0.0.1:2380 [default: %(default)s]'
    )
    parser.add_argument(
        '--db_passwd',
        type=str,
        default=None,
        help='detabase password [default: %(default)s]')
    parser.add_argument(
        '--db_type',
        type=str,
        default='etcd',
        help='database type, only support etcd for now [default: %(default)s]')
    parser.add_argument(
        '--service_name',
        type=str,
        help='service name where the server is located',
        required=True)
    parser.add_argument(
        '--server',
        type=str,
        help='endpoint of the server, e.g. 127.0.0.1:8888',
        required=True)
    # TODO. service_token
    parser.add_argument(
        '--service_token',
        type=str,
        default=None,
        help='service token, which the same can register [default: %(default)s]'
    )

    args = parser.parse_args()
    db_endpoints = args.db_endpoints.split(',')

    register = ServerRegister(db_endpoints, args.db_passwd, args.db_type)
    register.register(args.service_name, args.server)
