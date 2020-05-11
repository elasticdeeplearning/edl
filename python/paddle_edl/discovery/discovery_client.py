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

import grpc
import discovery_pb2
import discovery_pb2_grpc
import logging
import os
import threading
import time

from server_alive import is_server_alive

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class DiscoveryClient(object):
    def __init__(self, discover, service_name, require_num, token=None):
        self._channel = None
        self._stub = None
        self._endpoint = discover

        self._client = None
        self._service_name = service_name
        self._require_num = require_num
        self._token = token

        self._version = 0
        self._ret_servers = []

        self._beat_thread = None
        self._stop_event = threading.Event()

    def _process_response(self, response):
        # TODO.
        msg = response.msg
        version = response.version
        if version > self._version:
            self._ret_servers = response.servers
            self._version = version

        logging.debug('response, msg={} version={} servers={}'.format(
            response.msg, response.version, response.servers))

    def _register(self):
        register_request = discovery_pb2.RegisterRequest(
            client=self._client,
            service_name=self._service_name,
            require_num=self._require_num,
            token=self._token)
        response = self._stub.Register(register_request)
        self._process_response(response)
        logging.debug(
            'register client={} service_name={} require_num={} token={}'.
            format(self._client, self._service_name, self._require_num,
                   self._token))

    def _heartbeat(self):
        is_register = False
        while not self._stop_event.is_set():
            if not is_register:
                self._register()
                is_register = True

            beat_request = discovery_pb2.HeartBeatRequest(
                client=self._client, version=self._version)
            response = self._stub.HeartBeat(beat_request)
            self._process_response(response)
            time.sleep(2)

    def start(self, daemon=True):
        if self._channel is not None:
            return

        failed_count = 0
        max_failed_count = 10
        while True:
            alive, client_addr = is_server_alive(self._endpoint)
            if alive:
                break
            failed_count += 1
            logging.warning('discovery server={} is not alive, failed_count={}'
                            .format(self._endpoint, failed_count))
            if failed_count == max_failed_count:
                raise Exception('discovery server is not alive')
            time.sleep(1)

        self._channel = grpc.insecure_channel(self._endpoint)
        self._stub = discovery_pb2_grpc.DiscoveryServiceStub(self._channel)

        ip = client_addr[0]
        pid = os.getpid()
        sid = hex(id(self._channel))
        time_stamp = int(time.time() * 1000)
        # FIXME. client_uuid=ip-pid-_channel_id-timestamp, need a better method?
        self._client = '{}-{}-{}-{}'.format(ip, pid, sid, time_stamp)

        self._beat_thread = threading.Thread(target=self._heartbeat)
        self._beat_thread.daemon = daemon
        self._beat_thread.start()

    def stop(self):
        if self._channel is None:
            return
        self._stop_event.set()
        self._beat_thread.join()
        self._stop_event.clear()

        self._stub = None
        self._channel.close()

    def get_servers(self):
        return self._ret_servers


if __name__ == '__main__':
    client = DiscoveryClient('127.0.0.1:50051', 'TestService', 4)
    client.start(daemon=True)
    for i in range(1000):
        servers = client.get_servers()
        print(servers)
        time.sleep(1)
