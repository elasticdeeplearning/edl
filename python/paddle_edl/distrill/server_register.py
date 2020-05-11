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
import socket
import time


class ServerRegister(object):
    def __init__(self, ip, port, service_name, store):
        self._ip = ip
        self._port = port
        self._server = ip + ':' + str(port)
        self._service_name = service_name
        self._store = store
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _get_info(self):
        # Todo
        info = '{cpu:10%, gpu:20%, net:1}'
        return info

    def _register(self, ttl=180):
        while not self._is_alive() and ttl > 0:
            print("start to register, but port is not open, ttl=" + str(ttl))
            ttl -= 2
            time.sleep(3)

        if ttl <= 0:
            raise
        self._store.set_server(self._service_name, self._server,
                               self._get_info())
        print("register success")

    def _is_alive(self):
        alive = True
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect((self._ip, self._port))
            s.shutdown(socket.SHUT_RDWR)
            alive = True
        except:
            alive = False
        finally:
            s.close()

        return alive

    def _heartbeat(self, beat_time=1.5):
        retry = 60
        failed_count = 0
        while failed_count < retry:
            while self._is_alive():
                if failed_count != 0:
                    self._store.set_server(self._service_name, self._server,
                                           self._get_info())
                ret = self._store.refresh(self._service_name, self._server)
                if ret is False:
                    break
                failed_count = 0
                time.sleep(beat_time)
            failed_count += 1
            print("server is not alive, retry=" + str(failed_count))
            time.sleep(2)
        print("retry timeout, exit")

    def start(self, daemon=False):
        self._register()
        self._heartbeat()
        # self._thread = threading.Thread(self._heartbeat())
        # self._thread.daemon = daemon
        # self._thread.start()


if __name__ == '__main__':
    import sys
    from redis_store import RedisStore

    ip = '127.0.0.1'
    port = 5458
    service_name = 'TestService'
    service_name = 'DistillService'

    if len(sys.argv) == 2:
        port = int(sys.argv[1])
    elif len(sys.argv) >= 3:
        ip = sys.argv[1]
        port = int(sys.argv[2])
        if len(sys.argv) == 4:
            service_name = sys.argv[3]

    print('register {}:{} service_name={}'.format(ip, port, service_name))

    #store = RedisStore('127.0.0.1', 6379)
    store = RedisStore('10.255.100.13', 6379)
    register = ServerRegister(ip, port, service_name, store)
    register.start()
