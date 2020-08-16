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
import select
import struct
import errno
import json
import threading
import sys
from six.moves.queue import Queue
import time


class Server(object):
    _READ = select.EPOLLIN | select.EPOLLHUP | select.EPOLLERR
    _WRITE = select.EPOLLOUT | select.EPOLLHUP | select.EPOLLERR
    RECV_SIZE = 4096
    HEAD_SIZE = 8  # 8bytes, 64bit
    HEAD_FORMAT = '!4si'
    CRC_CODE = b'\xCB\xEF\x00\x00'

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._clients = {}

        # request & response message
        self._requests = {}
        self._responses = {}
        self._request_queue = Queue()
        self._response_queue = Queue()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setblocking(False)
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._ip, self._port))
        server.listen(5)
        fd = server.fileno()

        self._server = server
        self._fd = fd

        # create epoll
        epoll = select.epoll()
        epoll.register(fd, self._READ)
        self._epoll = epoll

    def _process_msg(self, fd, msg):
        pass

    def _handle_requests(self):
        while True:
            fd, msg = self._request_queue.get()
            crc_code, length = struct.unpack_from(self.HEAD_FORMAT, msg)
            if length != len(msg):
                sys.stderr.write('length={} != msg_length={}, close\n'.format(
                    length, len(msg)))
                self.close_conn(fd)
                continue
            msg = json.loads(msg[self.HEAD_SIZE:].decode())
            self._process_msg(fd, msg)

    def _enqueue_request(self, fd):
        msg = self._requests[fd]
        if len(msg) < self.HEAD_SIZE:
            return

        while True:
            crc_code, length = struct.unpack_from(self.HEAD_FORMAT, msg)
            if crc_code != self.CRC_CODE:
                # connection error
                # self._epoll.modify(fd, select.EPOLLERR)
                self.close_conn(fd)
                return

            if len(msg) < length:
                return

            request = msg[:length]
            self._request_queue.put((fd, request))

            self._requests[fd] = msg[length:]
            msg = self._requests[fd]
            if len(msg) < self.HEAD_SIZE:
                return

    def _enqueue_response(self, fd, msg):
        msg = json.dumps(msg).encode()
        size = self.HEAD_SIZE + len(msg)

        msg = struct.pack(self.HEAD_FORMAT, self.CRC_CODE, size) + msg
        assert len(msg) == size, 'Error with response msg'
        # self._response_queue.put((fd, msg))

        # Fixme. response multi msg?
        # assert len(self._responses[fd]) == 0
        self._responses[fd] += msg
        self._epoll.modify(fd, self._WRITE)

    def _init_conn(self):
        client, addr = self._server.accept()
        # sys.stderr.write('addr={} conn\n'.format(addr))
        # client.getpeername()

        client.setblocking(False)
        fd = client.fileno()
        self._epoll.register(fd, self._READ)
        self._clients[fd] = client
        self._requests[fd] = b''
        self._responses[fd] = b''

    def _handle_in(self, fd):
        try:
            data = self._clients[fd].recv(self.RECV_SIZE)
        except socket.error as e:
            eno = e.args[0]
            if eno not in (errno.EINTR, errno.EWOULDBLOCK, errno.EAGAIN):
                # connection error
                # self._epoll.modify(fd, select.EPOLLERR)
                self.close_conn(fd)
            return
        if not data:
            # connection close
            # self._epoll.modify(fd, select.EPOLLHUP)
            self.close_conn(fd)
            return
        else:
            self._requests[fd] += data

        self._enqueue_request(fd)

    def _handle_out(self, fd):
        response = self._responses[fd]
        size = len(response)
        try:
            send_size = self._clients[fd].send(response)
        except socket.error as e:
            eno = e.args[0]
            if eno not in (errno.EINTR, errno.EWOULDBLOCK, errno.EAGAIN):
                # connection error
                # self._epoll.modify(fd, select.EPOLLERR)
                self.close_conn(fd)
            return
        if send_size == 0:
            # connection close
            # self._epoll.modify(fd, select.EPOLLHUP)
            self.close_conn(fd)
            return
        else:
            self._responses[fd] = response[send_size:]
            if len(self._responses[fd]) == 0:
                self._epoll.modify(fd, self._READ)

    def close_conn(self, fd):
        try:
            ip, port = self._clients[fd].getpeername()
            sys.stderr.write('close conn={}\n'.format(ip + ':' + str(port)))
            self._epoll.unregister(fd)
            self._clients[fd].close()
        except Exception as e:
            sys.stderr.write('Exception when close fd={}\n'.format(fd))
            sys.stderr.write(str(e) + '\n')
        del self._clients[fd]
        del self._requests[fd]
        del self._responses[fd]

    def _start(self):
        request_thread = threading.Thread(target=self._handle_requests)
        request_thread.daemon = True
        request_thread.start()
        # Todo. Add response thread?
        while True:
            for fd, event in self._epoll.poll(timeout=1):
                if fd == self._fd:
                    self._init_conn()
                elif (event & select.EPOLLHUP) or (event & select.EPOLLERR):
                    self.close_conn(fd)
                elif event & select.EPOLLIN:
                    self._handle_in(fd)
                elif event & select.EPOLLOUT:
                    self._handle_out(fd)

    def server_forever(self):
        try:
            self._start()
        finally:
            self._epoll.unregister(self._fd)
            self._epoll.close()
            self._server.close()


class BalanceServer(Server):
    def __init__(self, ip='127.0.0.1', port=9379, table=None):
        super(BalanceServer, self).__init__(ip, port)
        self._table = table
        self._handle_func = {
            'register': self._handle_register,
            'heartbeat': self._handle_heartbeat
        }

    def _handle_register(self, fd, msg):
        # Todo
        # store.set_client()
        require_num = int(msg['num'])
        self._table.add_service_name(fd, msg['service_name'], require_num)
        servers = self._table.get_servers(fd, require_num)

        client = self._clients[fd]
        ip, port = client.getpeername()
        sys.stderr.write('register addr={} service_name={} num={}\n'.format(
            ip + ':' + str(port), msg['service_name'], require_num))

        # response
        msg = {
            'type': 'register',
            'seq': int(msg['seq']) + 1,
            'servers': servers,
            'num': len(servers)
        }
        self._enqueue_response(fd, msg)

    def _handle_heartbeat(self, fd, msg):
        version = 0
        try:
            version = int(msg['version'])
        except KeyError:
            # compatible old client
            pass
        new_version, servers = self._table.is_servers_update(fd, version)
        if new_version > version:
            msg = {
                'type': 'servers_change',
                'servers': servers,
                'version': new_version
            }
        else:
            msg = {'type': 'heartbeat'}
        self._enqueue_response(fd, msg)

    def _process_msg(self, fd, msg):
        type = msg['type']
        func = self._handle_func[type]
        func(fd, msg)

    def close_conn(self, fd):
        super(BalanceServer, self).close_conn(fd)
        self._table.rm_service_name(fd)

    def server_forever(self):
        self._table.start()
        super(BalanceServer, self).server_forever()


if __name__ == '__main__':
    from .service_table import ServiceTable

    import argparse
    parser = argparse.ArgumentParser(
        description='Discovery server with balance')
    parser.add_argument(
        '--server',
        type=str,
        default='0.0.0.0:7001',
        help='endpoint of the server, e.g. 127.0.0.1:8888 [default: %(default)s]'
    )
    parser.add_argument(
        '--worker_num',
        type=int,
        default=1,
        help='worker num of server [default: %(default)s]')
    parser.add_argument(
        '--db_endpoints',
        type=str,
        default='127.0.0.1:6379',
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
        default='redis',
        help='database type, only support redis for now [default: %(default)s]')

    args = parser.parse_args()
    server = args.server
    worker_num = args.worker_num
    db_endpoints = args.db_endpoints.split(',')

    redis_ip_port = db_endpoints[0].split(':')
    server_ip_port = server.split(':')

    table = ServiceTable(redis_ip_port[0],
                         int(redis_ip_port[1]))  # connect redis ip:port
    balance_server = BalanceServer(server_ip_port[0],
                                   int(server_ip_port[1]), table)  # listen
    balance_server.server_forever()
