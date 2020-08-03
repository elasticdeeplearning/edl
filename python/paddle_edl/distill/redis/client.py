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

import socket
import threading
import json
import struct
import sys
import time
import os


class Client(object):
    RECV_SIZE = 4096
    HEAD_SIZE = 8  # 8bytes, 64bit
    HEAD_FORMAT = '!4si'
    CRC_CODE = b'\xCB\xEF\x00\x00'

    def __init__(self, endpoints, service_name, require_num, token=None):
        """
        Args:
            endpoints(list|tuple): BalanceServer endpoints
            service_name(str): service name
            require_num(int): require max teacher
            token(str):
        """
        ip, port = endpoints[0].split(':')
        self._ip = ip
        self._port = int(port)

        self._service_name = service_name
        self._require_num = require_num
        self._token = token
        self._need_stop = False

        self._balance_list = endpoints
        self.teacher_list = []
        self._is_update = False
        self._version = 0

        self._thread = None
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _recv_msg(self):
        head = self.client.recv(self.HEAD_SIZE)
        crc_code, length = struct.unpack(self.HEAD_FORMAT, head)
        if crc_code != self.CRC_CODE:
            # error Todo
            assert False

        data = self.client.recv(length - self.HEAD_SIZE)
        msg = json.loads(data.decode())
        return msg

    def _send_msg(self, msg):
        msg = json.dumps(msg).encode()
        size = self.HEAD_SIZE + len(msg)

        msg = struct.pack(self.HEAD_FORMAT, self.CRC_CODE, size) + msg
        assert len(msg) == size
        self.client.sendall(msg)

    def _register(self, require_num=1):
        seq = 0
        msg = {
            'type': 'register',
            'service_name': self._service_name,
            'seq': seq,
            'num': require_num
        }
        self._send_msg(msg)
        msg = self._recv_msg()
        if msg['type'] != 'register' or int(msg['seq']) != seq + 1:
            assert False

        response_num = msg['num']
        servers = msg['servers']
        return servers

    def _heartbeat(self):
        while self._need_stop is False:
            time.sleep(2)
            msg = {'type': 'heartbeat', 'version': self._version}
            self._send_msg(msg)
            msg = self._recv_msg()

            if msg['type'] == 'heartbeat':
                #print('heartbeat')
                continue
            elif msg['type'] == 'servers_change':
                self._is_update = True
                try:
                    self._version = int(msg['version'])
                except KeyError:
                    # compatible with old balance server
                    pass
                self.teacher_list = msg['servers']
                sys.stderr.write(
                    '[INFO] service change version={} teachers={}\n'.format(
                        self._version, str(self.teacher_list)))

    def get_teacher_list(self):
        '''
        return (is_update, servers)
        is_update: is update after last query
        '''
        is_update = self._is_update
        self._is_update = False
        return is_update, self.teacher_list

    def get_servers(self):
        return self.teacher_list

    def start(self, daemon=True):
        try:
            self.client.connect((self._ip, self._port))
        except socket.error:
            sys.stderr.write('ERROR: connect with balance server failed, '
                             'please check if balance server is alive\n')
            os._exit(-1)
        # self.client.settimeout(6)

        self.teacher_list = self._register(self._require_num)
        self._thread = threading.Thread(target=self._heartbeat)
        self._thread.daemon = True
        self._need_stop = False
        self._thread.start()
        return self.teacher_list

    def stop(self):
        self._need_stop = True
        self._thread.join()
        self.client.shutdown(socket.SHUT_RDWR)
        self.client.close()


if __name__ == '__main__':
    client = Client(['127.0.0.1:9379'], 'TestService', 4)
    teacher_list = client.start()
    print(teacher_list)
    time.sleep(100)
    print(client.get_teacher_list())
    client.stop()
