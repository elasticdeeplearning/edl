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

import functools
import logging
import threading
import time
import sys

from collections import deque
from etcd_client import EtcdClient
from six.moves import queue
from weakref import WeakValueDictionary

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class Service(object):
    def __init__(self, name, update_event_callback):
        """ name: service_name
            update_event_callback: callback(name), representative service needs to be rebalanced
        """
        # TODO. write this to db?
        self.name = name
        self.update_event_callback = functools.partial(update_event_callback,
                                                       name)

        self.reference_count = 0
        self.service_mutex = threading.Lock()  # mutex

        self.client_update = False  # if servers or clients change, need update

        # get from db
        self.get_mutex = threading.Lock()
        self.db_servers = None  # get from db, [[server, info], ...]

        self.servers = set()  # servers in service, cache from store
        # clients served by server {server: set(client), }
        self.server_to_clients = dict()

        self.clients = set()  # clients connect with service
        # servers connected by client {client: set(servers), }
        self.client_to_servers = dict()
        # client to servers set version {client: 0, }
        self.client_to_version = dict()
        # maximum number of servers required by client
        self.client_to_maxn = dict()

        # for watch or subscribe
        self.watch_id = None
        self.watch_mutex = threading.Lock()
        self.add_servers = dict()
        self.rm_servers = set()

    def _need_update(self):
        self.update_event_callback()

    def inc_ref(self):
        assert self.reference_count >= 0, \
            'reference count of service={} must >= 0'.format(self.name)
        self.reference_count += 1

    def dec_ref(self):
        assert self.reference_count > 0, \
            'when decrease, reference count of service={} must > 0'.format(self.name)
        self.reference_count -= 1
        return self.reference_count

    def add_client(self, client, require_num):
        with self.service_mutex:
            self.clients.add(client)
            self.client_to_maxn[client] = require_num
            self.client_to_version[client] = 0
            self.client_to_servers[client] = set()
            self.client_update = True

        self._need_update()

    def remove_client(self, client):
        with self.service_mutex:
            self.clients.remove(client)
            self.client_to_maxn.pop(client)
            self.client_to_version.pop(client)

            # remove client from server
            for server in self.client_to_servers[client]:
                self.server_to_clients[server].remove(client)
            self.client_to_servers.pop(client)

            self.client_update = True

        self._need_update()

    def watch_call_back(self, add_servers, rm_servers):
        need_update = True
        with self.watch_mutex:
            # after rm key, add again, so need remove from rm set
            # self.rm_servers -= set(add_servers.keys())
            self.rm_servers.difference_update(add_servers.keys())
            # after add key, rm again, so need remove from add dict
            map(lambda x: self.add_servers.pop(x, None), rm_servers)

            # update add_servers & rm_servers
            self.add_servers.update(add_servers)
            self.rm_servers.update(rm_servers)

            if len(self.add_servers) == 0 and len(self.rm_servers) == 0:
                need_update = False

        if need_update:
            self._need_update()

    def set_servers(self, servers):
        with self.get_mutex:
            self.db_servers = servers
        self._need_update()

    def rebalance(self):
        # The service includes the following changes:
        # 1. Initially or periodically obtain servers from db to ensure
        # consistency with the database.
        # 2. The increase and decrease of the servers observed in the watch.
        # 3. The increase and decrease of the clients.
        # TODO. need revision?

        db_servers = None
        with self.get_mutex:
            self.db_servers, db_servers = db_servers, self.db_servers

        add_servers = dict()
        rm_servers = set()
        with self.watch_mutex:
            self.add_servers, add_servers = add_servers, self.add_servers
            self.rm_servers, rm_servers = rm_servers, self.rm_servers

        if db_servers is not None:
            servers = set([s[0] for s in db_servers])
            # FIXME. add_servers or rm_servers may ahead or behind with
            # servers, but it doesn't matter?
            # 1. If add_servers is ahead, it's ok;
            # 2. If add_servers is behind, info is old, but will update next
            # watch event(for revision is behind, will be watched next time)
            # or period db refresh.
            # 3. If rm_servers is ahead, it's ok;
            # 4. If rm_servers is behind, del null, I think it's ok.
            servers.update(add_servers.keys())
            servers -= rm_servers

            old_servers = self.servers
            self.servers = servers  # update servers

            rm_servers = set(old_servers) - set(servers)
            add_servers = set(servers) - set(old_servers)
        else:  # only watch
            rm_servers = rm_servers
            add_servers = set(add_servers.keys())

            # update servers
            # self.servers -= rm_servers
            self.servers.difference_update(rm_servers)
            self.servers.update(add_servers)
            servers = self.servers

        with self.service_mutex:
            # no change
            if len(rm_servers) == 0 and len(add_servers) == 0 and \
                    self.client_update is False:
                return

            self.client_update = False

            # client to servers is change
            update_client = set()

            # remove servers
            for server in rm_servers:
                if server not in self.server_to_clients:
                    logging.warning(
                        'service={}, when remove server, server={} not in server_to_clients'.
                        format(self.name, server))
                    continue
                # remove servers which connected by client
                for client in self.server_to_clients[server]:
                    try:
                        self.client_to_servers[client].remove(server)
                        # client is updated
                        update_client.add(client)
                    except KeyError:  # it's must be impossible
                        logging.critical(
                            "service={} when remove server, server={} in server_to_clients, "
                            "but not in client_to_servers with client={}, it's "
                            'must be impossible, something maybe wrong?'.
                            format(self.name, server, client))
                # remove server from server_to_clients
                self.server_to_clients.pop(server)

            client_num = len(self.clients)
            server_num = len(self.servers)

            # all client may be unregister, but client_to_service or
            # name_to_service hasn't clean yet
            if client_num == 0:
                assert len(update_client) == 0, "if all client unregister, len of" \
                                                "update_client must == 0"
            # no servers in service =.=
            if server_num == 0:
                logging.warning('service={} have no servers'.format(self.name))
                for client in update_client:
                    self.client_to_version[client] += 1
                return

            # rebalance. TODO. need a better algorithm. e.g.
            # Consistency hash algorithm, Network flow algorithm?

            # tmp use following method.
            # 1. assume: client_num=3, server_num=97
            # each server can provide max=1 conn
            # each client can connect max=32 servers
            # assign: {client0:32, client1:32, client2:32}
            # 2. assume: client_num=97, server_num=3
            # each server can provide max=33 conn
            # each client can connect max=1 servers
            # assign: {server0:33, server1:33, server2:31)
            max_conn_provided_by_server = int(
                (client_num + server_num - 1) / server_num)
            max_servers_client_can_conn = max(1, int(server_num / client_num))
            logging.info(
                '<Global> service={}, client_num={}, server_num={}, '
                'server_provided_conn={}, client_can_conn={}'.format(
                    self.name, client_num, server_num,
                    max_conn_provided_by_server, max_servers_client_can_conn))

            # limit connect provided by server
            for server in servers:
                conn_clients = self.server_to_clients.setdefault(server, set())
                while len(conn_clients) > max_conn_provided_by_server:
                    # break link
                    client = conn_clients.pop()
                    self.client_to_servers[client].remove(server)

                    # client conn is update
                    update_client.add(client)
                    logging.info(
                        'service={} break link with server={} -> client={}'.
                        format(self.name, server, client))

            # client greedy connect with server
            for client in self.clients:
                max_connect = min(max_servers_client_can_conn,
                                  self.client_to_maxn[client])
                logging.info('<Client> service={} client={} max_connect={}'.
                             format(self.name, client, max_connect))

                conn_servers = self.client_to_servers[client]
                # limit servers connected by client
                while len(conn_servers) > max_connect:
                    # break link
                    server = conn_servers.pop()
                    self.server_to_clients[server].remove(client)

                    # client conn is update
                    update_client.add(client)
                    logging.info(
                        '<Breaking> service={} client-/->server {} -/-> {}'.
                        format(self.name, client, server))

                # TODO. need optimize
                for server in servers:
                    # client connect enough servers
                    if len(conn_servers) >= max_connect:
                        break
                    # server is already connected with client
                    if server in conn_servers:
                        continue
                    # server already provided max conn
                    if len(self.server_to_clients[
                            server]) >= max_conn_provided_by_server:
                        continue

                    conn_servers.add(server)
                    self.server_to_clients[server].add(client)

                    # client conn is update
                    update_client.add(client)
                    logging.info(
                        '<Linking> service={} client-->server {} --> {}'.
                        format(self.name, client, server))

            for client in update_client:
                self.client_to_version[client] += 1

    def get_servers(self, client, version):
        """ external service interface """
        new_version = self.client_to_version[client]

        if new_version > version:  # is update
            return new_version, list(self.client_to_servers[client])
        else:
            return new_version, None


class Entry(object):
    def __init__(self, client, table):
        self._client = client
        self._table = table

    def __del__(self):
        self._table.unregister_client(self._client)


class BalanceTable(object):
    def __init__(self,
                 db_endpoints,
                 db_passwd=None,
                 db_type='etcd',
                 idle_seconds=7):
        self._db = EtcdClient(db_endpoints, db_passwd)
        self._is_db_connect = False

        self._mutex = threading.Lock()

        self._client_to_service = dict()
        self._name_to_service = dict()

        self._get_service_thread = None
        self._new_service_queue = queue.Queue()

        self._update_service_thread = None
        self._update_event_queue = queue.Queue()

        # timing wheel, unregister client
        self._client_timing_buckets = deque(maxlen=idle_seconds)
        for _ in range(idle_seconds):
            self._client_timing_buckets.append(list())

        self._client_weak_entrys = WeakValueDictionary()

    def _add_update_event(self, name):
        self._update_event_queue.put(name)

    def _get_service_task(self, service):
        # [[server, info], ...]
        servers = self._db.get_service(service.name)
        service.set_servers(servers)

    def _period_get_service_worker(self, period=20):
        """ period get all service from db """
        while True:
            # name_to_service may be out-of-data, e.g. new service be added and
            # old service be removed after we get, but if doesn't matter.
            for service in self._name_to_service.values():
                # FIXME. use thread pool, is db support?
                self._get_service_task(service)

            try:
                timeout = period
                end_time = time.time() + timeout
                while True:
                    # get service immediately. NOTE, pre for maybe already get this
                    # new service, but it doesn't matter
                    new_service = self._new_service_queue.get(timeout=timeout)
                    self._get_service_task(new_service)

                    timeout = end_time - time.time()
                    if timeout <= 0:
                        break
                    else:
                        end_time = time.time() + timeout
            except queue.Empty:
                pass

    def _update_service_worker(self, timing_wheel=1):
        import gc
        while True:
            try:
                timeout = timing_wheel
                end_time = time.time() + timeout
                while True:
                    name = self._update_event_queue.get(timeout=timeout)
                    try:
                        service = self._name_to_service[name]
                        # TODO. add thread pool? If use thread pool. A service can
                        # only be updated by one thread at the same time
                        service.rebalance()
                    except KeyError:
                        pass

                    timeout = end_time - time.time()
                    if timeout <= 0:
                        break
                    else:
                        end_time = time.time() + timeout
            except queue.Empty:
                pass

            self._client_timing_buckets.append(list())
            # make sure entry __del__ exec
            gc.collect()

    def _register_watch_service(self, service_name, callback):
        def _call_back(response):
            add_servers, rm_servers = self._db.services_change(response,
                                                               service_name)
            if len(add_servers) == 0 and len(rm_servers) == 0:
                return
            else:
                callback(add_servers, rm_servers)

        watch_id = self._db.watch_service(service_name, _call_back)
        return watch_id

    def _unregister_watch_service(self, watch_id):
        self._db.cancel_watch(watch_id)

    def register_client(self, client, service_name, require_num, token=None):
        with self._mutex:
            if client in self._client_to_service:
                # already registered
                if self._client_to_service[client].name == service_name:
                    logging.warning(
                        'client={} register again service_name={} require_num={}'.
                        format(client, service_name, require_num))
                    return
                else:  # NOTE. This must be impossible
                    logging.critical(
                        'client={} register new service_name={} require_num={}, this is impossible'
                        .format(client, service_name, require_num))
                    return

            if service_name in self._name_to_service:
                service = self._name_to_service[service_name]
            else:
                # add new service monitor
                service = Service(service_name, self._add_update_event)
                self._new_service_queue.put(service)

                self._name_to_service[service_name] = service
                # save watch_id in service
                service.watch_id = self._register_watch_service(
                    service_name, service.watch_call_back)

            self._client_to_service[client] = service
            service.inc_ref()

        service.add_client(client, require_num)

        # timing wheel
        entry = Entry(client, table=self)
        self._client_timing_buckets[-1].append(entry)
        self._client_weak_entrys[client] = entry

        logging.info('register client={} service_name={} require_num={}'.
                     format(client, service_name, require_num))

    def unregister_client(self, client):
        # timing wheel maybe unregister again
        if client not in self._client_to_service:
            return

        service = self._client_to_service[client]
        service.remove_client(client)

        with self._mutex:
            self._client_to_service.pop(client)

            service_ref_count = service.dec_ref()
            assert service_ref_count >= 0, \
                'service_ref_count must >=0, but true value={}'.format(service_ref_count)

            if service_ref_count == 0:  # remove service monitor
                self._unregister_watch_service(service.watch_id)
                self._name_to_service.pop(service.name)

        logging.info('unregister client={} service_name={}'.format(
            client, service.name))

        if service_ref_count == 0:
            logging.info('remove service_name={} monitor'.format(service.name))

    def get_servers(self, client, version):
        # timing wheel
        entry = self._client_weak_entrys.get(client)
        if entry is None:
            logging.warning('client={} timeout'.format(client))
            return None
        else:
            self._client_timing_buckets[-1].append(entry)
        return self._client_to_service[client].get_servers(client, version)

    def start(self):
        # start db
        self._db.init()
        self._get_service_thread = threading.Thread(
            target=self._period_get_service_worker)
        self._get_service_thread.daemon = True
        self._get_service_thread.start()

        self._update_service_thread = threading.Thread(
            target=self._update_service_worker)
        self._update_service_thread.daemon = True
        self._update_service_thread.start()
