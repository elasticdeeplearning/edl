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
import threading
import time
import sys
from .redis_store import RedisStore

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class ServiceTable(object):
    def __init__(self, ip='127.0.0.1', port=6379, passwd=None,
                 backend='redis'):
        if backend is 'redis':
            self._store = RedisStore(ip, port, passwd)
        elif backend is 'etcd':
            # Todo
            self._store = RedisStore(ip, port, passwd)
        self._fd_to_service_name = {}
        # service_name to set(fd)
        self._service_name_to_fds = {}
        # Todo. change to ReadWrite Lock. {service_name: lock}
        # self._service_name_to_fds mutex
        self._mutex = threading.RLock()

        self._service_name_to_update = {}

        # cache from store
        self._service_name_to_servers = {}

        # Todo. Balance assign
        # {fd: set(servers), }
        self._fd_to_servers = {}
        self._server_to_fds = {}
        self._fd_to_version = {}
        self._fd_to_max_num = {}

    def is_servers_update(self, fd, version):
        new_version = self._fd_to_version[fd]

        if new_version > version:  # is update
            return new_version, list(self._fd_to_servers[fd])
        else:
            return new_version, None

    def get_servers(self, fd, num):
        if fd not in self._fd_to_service_name:
            # not register
            return []
        service_name = self._fd_to_service_name[fd]

        if service_name not in self._service_name_to_servers or \
           self._service_name_to_update[service_name] is True:
            self._refresh_service(service_name)

        return list(self._fd_to_servers[fd])

    def add_service_name(self, fd, service_name, num):
        self._fd_to_servers[fd] = set()
        self._fd_to_version[fd] = 0
        self._fd_to_max_num[fd] = num
        print('fd={}, service_name={}, max_num={}'.format(fd, service_name,
                                                          num))
        self._fd_to_service_name[fd] = service_name
        with self._mutex:
            if service_name not in self._service_name_to_fds:
                self._service_name_to_fds[service_name] = {fd}
            else:
                self._service_name_to_fds[service_name].add(fd)
        self._service_name_to_update[service_name] = True

    def rm_service_name(self, fd):
        # client maybe exit before register
        if fd not in self._fd_to_service_name:
            return

        service_name = self._fd_to_service_name[fd]

        with self._mutex:
            if service_name in self._service_name_to_fds:
                try:
                    self._service_name_to_fds[service_name].remove(fd)
                except KeyError:
                    pass
                finally:
                    if len(self._service_name_to_fds[service_name]) == 0:
                        del self._service_name_to_fds[service_name]
                        del self._service_name_to_update[service_name]
                    else:
                        self._service_name_to_update[service_name] = True

        del self._fd_to_service_name[fd]

        del self._fd_to_max_num[fd]
        del self._fd_to_version[fd]

        for server in self._fd_to_servers[fd]:
            self._server_to_fds[server].remove(fd)
        del self._fd_to_servers[fd]

    def _refresh_service(self, service_name):
        if service_name not in self._service_name_to_servers:
            old_servers = []
        else:
            # list [ip_port0, ip_port1, ..., ]
            old_servers = self._service_name_to_servers[service_name]

        # list [ {'info': info, 'server': ip_port}, ..., }
        server_infos = self._store.get_service(service_name)

        # list [ip_port, ..., ]. Todo. info
        servers = []
        for s in server_infos:
            try:
                server = s['server']
                servers.append(server)
            except KeyError:
                # when get service, server may expired
                continue

        # update servers in service_name
        self._service_name_to_servers[service_name] = servers

        rm_servers = set(old_servers) - set(servers)
        add_servers = set(servers) - set(old_servers)

        # no change
        if len(rm_servers) == 0 and len(add_servers) == 0 and \
                self._service_name_to_update[service_name] is False:
            return
        self._service_name_to_update[service_name] = False
        update_fd = set()

        # remove server
        for server in rm_servers:
            if server not in self._server_to_fds:
                sys.stdout.write('{} not in server_to_fds'.format(server))
                continue
            # remove server in fd
            for fd in self._server_to_fds[server]:
                try:
                    self._fd_to_servers[fd].remove(server)
                    # update fd
                    update_fd.add(fd)
                except KeyError:
                    sys.stdout.write('{} not in fd={} servers'.format(server,
                                                                      fd))
            # remove server to fds
            del self._server_to_fds[server]

        # _service_name_to_fds maybe removed
        with self._mutex:
            if service_name not in self._service_name_to_fds:
                for fd in update_fd:
                    self._fd_to_version[fd] += 1
                return
            fd_num = len(self._service_name_to_fds[service_name])

        # print('fd_num={}'.format(fd_num))
        server_num = len(self._service_name_to_servers[service_name])
        if server_num == 0:
            print('service={} server_num=0'.format(service_name))
            for fd in update_fd:
                self._fd_to_version[fd] += 1
            return
        # assume: fd_num=3, server_num=97
        # assign: {fd0:32, fd1:32, fd2:32}
        server_max_connect = int((fd_num + server_num - 1) / server_num)
        fd_max_connect = max(1, int(server_num / fd_num))
        #fd_max_connect = int((server_num + fd_num - 1) / fd_num)
        print('fd_num={}, server_num={}, smax={}, mcon={}'.format(
            fd_num, server_num, server_max_connect, fd_max_connect))

        # server_conn = []  # [(num, server)]
        # rebalance
        # limit connect of server
        for server in servers:
            if server not in self._server_to_fds:
                self._server_to_fds[server] = set()
            while len(self._server_to_fds[server]) > server_max_connect:
                fd = self._server_to_fds[server].pop()
                self._fd_to_servers[fd].remove(server)
                update_fd.add(fd)
                print('pop fd={} server={}'.format(fd, server))
        try:
            fds = self._service_name_to_fds[service_name]
            for fd in fds:
                max_connect = min(fd_max_connect, self._fd_to_max_num[fd])
                logging.info('fd={} max_connect={}'.format(fd, max_connect))
                if fd not in self._fd_to_servers:
                    self._fd_to_servers[fd] = set()
                # limit connect of fd
                while len(self._fd_to_servers[fd]) > max_connect:
                    server = self._fd_to_servers[fd].pop()
                    self._server_to_fds[server].remove(fd)
                    update_fd.add(fd)
                    logging.info('pop1 fd={} server={}'.format(fd, server))

            # fd greed connect with server
            for fd in fds:
                max_connect = min(fd_max_connect, self._fd_to_max_num[fd])
                for server in servers:
                    if len(self._fd_to_servers[fd]) >= max_connect:
                        break
                    # have server or server connect is max, continue
                    if server in self._fd_to_servers[fd]:
                        continue
                    if len(self._server_to_fds[server]) >= server_max_connect:
                        continue
                    self._fd_to_servers[fd].add(server)
                    self._server_to_fds[server].add(fd)
                    update_fd.add(fd)
                    logging.info('add fd={} server={}'.format(fd, server))
        except Exception as e:
            sys.stderr.write(str(e) + '\n')

        for fd in update_fd:
            self._fd_to_version[fd] += 1

    def _refresh(self):
        while True:
            old_service_names = self._service_name_to_servers.keys()

            # Todo. ReadWrite Lock. Read Lock.
            # may be out-of-data, but if doesn't affect
            service_names = self._service_name_to_fds.keys()
            # refresh & add services
            for service_name in service_names:
                self._refresh_service(service_name)

            # rm service_name
            rm_service_names = set(old_service_names) - set(service_names)
            for service_name in rm_service_names:
                sys.stderr.write('Remove monitoring service={}\n'.format(
                    service_name))
                try:
                    del self._service_name_to_servers[service_name]
                except KeyError:
                    pass

            time.sleep(2)

    def start(self):
        self._thread = threading.Thread(target=self._refresh)
        self._thread.daemon = True
        self._thread.start()
