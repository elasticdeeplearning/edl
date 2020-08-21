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

from . import distill_discovery_pb2 as discovery
from . import distill_discovery_pb2_grpc
import functools
import grpc
import logging
import os
import random
import threading
import time

from ..discovery.server_alive import is_server_alive

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


def _handle_errors(f):
    def handler(*args, **kwargs):
        retry_times = 3
        for i in range(retry_times):
            if i > 0:
                args[0]._connect()
            try:
                return f(*args, **kwargs)
            except grpc.RpcError as e:
                logging.warning('grpc failed with {0}: {1}'.format(e.code(
                ), e.details()))

    return functools.wraps(f)(handler)


class DiscoveryClient(object):
    def __init__(self, endpoints, service_name, require_num, token=None):
        self._channel = None
        self._stub = None

        self._client = None
        self._service_name = service_name
        self._require_num = require_num
        self._token = token

        self._version = 0
        self._ret_servers = []

        self._discovery_version = 0
        self._discovery_servers = endpoints
        self._discover = None

        self._beat_thread = None
        self._stop_event = threading.Event()
        self._is_registered = False

        self._funcs = {
            discovery.Code.OK: self._process_ok,
            discovery.Code.UNKNOWN: self._error,
            discovery.Code.NO_READY: self._process_no_ready,
            discovery.Code.REDIRECT: self._process_redirect,
            discovery.Code.INVALID_ARGUMENT: self._error,
            discovery.Code.ALREADY_REGISTER: self._process_already_register,
            discovery.Code.REGISTER_OTHER_SERVICE: self._error,
            discovery.Code.UNREGISTERED: self._process_unregistered,
            discovery.Code.UNAUTHORIZED: self._error,
        }

    def _error(self, response):
        logging.error('client={} service={} error code={}'.format(
            self._client, self._service_name, response.status.code))
        assert False

    def _process_ok(self, response):
        if not self._is_registered:
            self._is_registered = True
            logging.debug('client={} register success'.format(self._client))

        if response.version > self._version:
            self._ret_servers = response.servers
            self._version = response.version
            logging.info('service version={} servers={}'.format(
                self._version, self._ret_servers))

        if response.discovery_version > self._discovery_version:
            self._discovery_servers = response.discovery_servers
            self._discovery_version = response.discovery_version
            logging.info('discovery_version={} servers={}'.format(
                self._discovery_version, self._discovery_servers))

    def _process_no_ready(self, response):
        logging.info('discovery server={} is not ready'.format(self._discover))
        pass

    def _process_redirect(self, response):
        self._is_registered = False

        old_discover = self._discover
        self._discover = response.status.message
        self._discovery_servers = response.discovery_servers
        self._discovery_version = response.discovery_version
        self._version = 0

        logging.info('redirect discovery server, old={} new={}'.format(
            old_discover, self._discover))

        # reconnect
        self._connect()

    def _process_already_register(self, response):
        logging.info('already register')
        pass

    def _process_unregistered(self, response):
        self._is_registered = False

    def _process_response(self, response):
        assert response.status.code in self._funcs
        self._funcs[response.status.code](response)

    @_handle_errors
    def _stub_register(self, register_request):
        return self._stub.Register(register_request)

    @_handle_errors
    def _stub_heartbeat(self, beat_request):
        return self._stub.HeartBeat(beat_request)

    def _register(self):
        register_request = discovery.RegisterRequest(
            client=self._client,
            service_name=self._service_name,
            require_num=self._require_num,
            token=self._token)

        logging.debug(
            'register client={} service_name={} require_num={} token={}'.
            format(self._client, self._service_name, self._require_num,
                   self._token))

        response = self._stub_register(register_request)
        self._process_response(response)

    def _heartbeat(self):
        while not self._stop_event.is_set():
            if not self._is_registered:
                self._register()

            beat_request = discovery.HeartBeatRequest(
                client=self._client,
                version=self._version,
                discovery_version=self._discovery_version)
            response = self._stub_heartbeat(beat_request)
            self._process_response(response)

            time.sleep(2)

    def _gen_client(self, addr, channel):
        ip = addr[0]
        pid = os.getpid()
        sid = hex(id(channel))
        time_stamp = int(time.time() * 1000)
        # FIXME. client_uuid=ip-pid-_channel_id-timestamp, need a better method?
        self._client = '{}-{}-{}-{}'.format(ip, pid, sid, time_stamp)

    def _connect_server(self, server):
        channel = None

        retry_times = 3
        for i in range(retry_times):
            alive, client_addr = is_server_alive(server)
            if alive:
                channel = grpc.insecure_channel(server)
                if self._client is None:
                    self._gen_client(client_addr, channel)
                break
            logging.warning(
                'discovery server={} is not alive, failed_count={}'.format(
                    server, i + 1))
            time.sleep(0.1 * (i + 1))

        return channel

    def _connect(self):
        # close pre channel
        if self._channel is not None:
            self._channel.close()

        channel = None
        if self._discover is not None:
            channel = self._connect_server(self._discover)

        if channel is None:
            endpoints = list(self._discovery_servers)
            random.shuffle(endpoints)

            for ep in endpoints:
                channel = self._connect_server(ep)
                if channel is not None:
                    break

        assert channel is not None, 'connect with discovery failed'
        self._channel = channel
        self._stub = distill_discovery_pb2_grpc.DiscoveryServiceStub(
            self._channel)

    def start(self, daemon=True):
        if self._channel is not None:
            return

        assert self._channel is None
        assert self._stub is None

        self._connect()
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
        self._channel = None

    def get_servers(self):
        return self._ret_servers


if __name__ == '__main__':
    client = DiscoveryClient(['127.0.0.1:50051', '127.0.0.1:50052'],
                             'TestService', 4)
    # client = DiscoveryClient(['127.0.0.1:50051'], 'TestService2', 4)
    client.start(daemon=True)
    for i in range(1000):
        servers = client.get_servers()
        print(servers)
        time.sleep(1)
