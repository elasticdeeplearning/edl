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
import functools
import logging
import threading
import time

from collections import deque
from paddle_edl.discovery.consistent_hash import ConsistentHash
from paddle_edl.discovery.etcd_client import EtcdClient
from paddle_edl.discovery.server_alive import is_server_alive
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
        self._servers_meta = None  # get from db, [[server, info], ...]

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
        self.add_servers = set()
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
        add_servers = [meta.server for meta in add_servers]
        rm_servers = [meta.server for meta in rm_servers]

        with self.watch_mutex:
            # after rm key, add again, so need remove from rm set
            # self.rm_servers -= set(add_servers.keys())
            self.rm_servers.difference_update(add_servers)
            # after add key, rm again, so need remove from add dict
            self.add_servers.difference_update(rm_servers)
            # map(lambda x: self.add_servers.pop(x, None), rm_servers)

            # update add_servers & rm_servers
            self.add_servers.update(add_servers)
            self.rm_servers.update(rm_servers)

            if len(self.add_servers) == 0 and len(self.rm_servers) == 0:
                need_update = False

        if need_update:
            self._need_update()

    def set_servers(self, servers_meta):
        with self.get_mutex:
            self._servers_meta = servers_meta
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
            self._servers_meta, db_servers = db_servers, self._servers_meta

        add_servers = set()
        rm_servers = set()
        with self.watch_mutex:
            self.add_servers, add_servers = add_servers, self.add_servers
            self.rm_servers, rm_servers = rm_servers, self.rm_servers

        if db_servers is not None:
            servers = set([meta.server for meta in db_servers])
            # FIXME. add_servers or rm_servers may ahead or behind with
            # servers, but it doesn't matter?
            # 1. If add_servers is ahead, it's ok;
            # 2. If add_servers is behind, info is old, but will update next
            # watch event(for revision is behind, will be watched next time)
            # or period db refresh.
            # 3. If rm_servers is ahead, it's ok;
            # 4. If rm_servers is behind, del null, I think it's ok.
            servers.update(add_servers)
            servers -= rm_servers

            old_servers = self.servers
            self.servers = servers  # update servers

            rm_servers = set(old_servers) - set(servers)
            add_servers = set(servers) - set(old_servers)
        else:  # only watch
            rm_servers = rm_servers
            add_servers = set(add_servers)

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
                return
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
                 discovery_server,
                 db_endpoints,
                 db_passwd=None,
                 db_type='etcd',
                 idle_seconds=7):
        self._discovery_server = discovery_server
        self._db = EtcdClient(db_endpoints, db_passwd)
        self._is_db_connect = False

        self._mutex = threading.Lock()

        self._client_to_service = dict()
        self._name_to_service = dict()

        self._consistent_hash = None
        self._consistent_hash_thread = None

        self._update_service_thread = None
        self._update_event_queue = queue.Queue()

        # timing wheel, unregister client
        self._client_timing_buckets = deque(maxlen=idle_seconds)
        for _ in range(idle_seconds):
            self._client_timing_buckets.append(list())

        self._client_weak_entrys = WeakValueDictionary()

    def _add_update_event(self, name):
        self._update_event_queue.put(name)

    def _consistent_hash_worker(self, ttl=120):
        all_time = ttl
        while not is_server_alive(self._discovery_server)[0] and ttl > 0:
            logging.warning(
                'start to register discovery server, but server is not start, ttl={}'.
                format(ttl))
            ttl -= 2
            time.sleep(2)

        if ttl <= 0:
            logging.error('discovery is not up in time={}s'.format(all_time))
            raise Exception('server up timeout')

        service_name = '__balance__'

        # register discovery server with balance to /service/__balance__/nodes/addr = ''
        self._db.set_server_not_exists(service_name, self._discovery_server,
                                       '')
        logging.info('register discovery server={} success'.format(
            self._discovery_server))

        servers_meta = self._db.get_service(service_name)
        logging.info('discovery service={}'.format((
            [str(server) for server in servers_meta])))

        servers = [meta.server for meta in servers_meta]
        assert self._discovery_server in servers  # must in discovery server

        self._consistent_hash = ConsistentHash(servers)

        revision = servers_meta[0].revision
        watch_queue = queue.Queue(100)  # Change must be infrequently

        def call_back(add_servers, rm_servers):
            if len(add_servers) == 0 and rm_servers == 0:
                return

            watch_queue.put((add_servers, rm_servers))

        self._db.refresh(service_name,
                         self._discovery_server)  # before watch, refresh
        # NOTE. start from revision + 1, that is after get_service
        watch_id = self._db.watch_service(
            service_name, call_back, start_revision=revision + 1)

        period = 2  # 2 seconds refresh
        while True:
            self._db.refresh(service_name, self._discovery_server)
            try:
                timeout = period
                end_time = time.time() + timeout
                while True:
                    server_change = watch_queue.get(timeout=timeout)
                    add_servers, rm_servers = server_change

                    for server_meta in rm_servers:
                        logging.info('Remove discovery server={}'.format(
                            server_meta))
                        self._consistent_hash.remove_node(server_meta.server)
                    for server_meta in add_servers:
                        logging.info('Add discovery server={}'.format(
                            server_meta))
                        self._consistent_hash.add_new_node(server_meta.server)

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

    def _unregister_watch_service(self, watch_id):
        self._db.cancel_watch(watch_id)

    def _get_service_from_db(self, service):
        # even if no server in service, revision will also return
        servers_meta, revision = self._db.get_service_with_revision(
            service.name)
        logging.info('get_service={}, servers={}, revision={}'.format(
            service.name, servers_meta, revision))
        if len(servers_meta) != 0:
            service.set_servers(servers_meta)
        # save watch_id into service. NOTE, watch from revision + 1
        service.watch_id = self._db.watch_service(
            service.name, service.watch_call_back, start_revision=revision + 1)

    def register_client(self, client, service_name, require_num, token=None):
        if self._consistent_hash is None:
            # return discovery server is not ready, client need retry
            status = discovery.Status(code=discovery.Code.NO_READY)
            return discovery.Response(status=status)

        # All discovery requests with the same service name are sent to the same server
        discovery_server, discovery_servers, discovery_version = \
            self._consistent_hash.get_node_nodes(service_name)
        if discovery_server != self._discovery_server:
            # request need sent to another discovery server
            status = discovery.Status(
                code=discovery.Code.REDIRECT, message=discovery_server)
            redirect_response = discovery.Response(
                status=status,
                discovery_version=discovery_version,
                discovery_servers=discovery_servers)
            return redirect_response

        is_new_service = False
        with self._mutex:
            if client in self._client_to_service:
                # already registered
                if self._client_to_service[client].name == service_name:
                    status = discovery.Status(
                        code=discovery.Code.ALREADY_REGISTER)
                    return discovery.Response(
                        status=status,
                        discovery_version=discovery_version,
                        discovery_servers=discovery_servers)
                else:  # NOTE. This must be impossible
                    logging.critical(
                        'client={} register new service_name={} require_num={}, this is impossible'
                        .format(client, service_name, require_num))
                    status = discovery.Status(
                        code=discovery.Code.REGISTER_OTHER_SERVICE)
                    return discovery.Response(status=status)

            if service_name in self._name_to_service:
                service = self._name_to_service[service_name]
            else:
                # add new service monitor
                service = Service(service_name, self._add_update_event)
                self._name_to_service[service_name] = service
                is_new_service = True

            self._client_to_service[client] = service
            service.inc_ref()

        if is_new_service:
            self._get_service_from_db(service)

        service.add_client(client, require_num)

        # timing wheel
        entry = Entry(client, table=self)
        self._client_timing_buckets[-1].append(entry)
        self._client_weak_entrys[client] = entry

        logging.info('register client={} service_name={} require_num={}'.
                     format(client, service_name, require_num))

        status = discovery.Status(code=discovery.Code.OK)
        return discovery.Response(
            status=status,
            discovery_version=discovery_version,
            discovery_servers=discovery_servers)

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
            logging.warning('client={} timeout or unregister'.format(client))
            status = discovery.Status(code=discovery.Code.UNREGISTERED)
            return discovery.Response(status=status)
        else:
            self._client_timing_buckets[-1].append(entry)

        service = self._client_to_service[client]
        service_name = service.name

        if self._consistent_hash is None:
            # return discovery server is not ready, client need retry
            status = discovery.Status(code=discovery.Code.NO_READY)
            return discovery.Response(status=status)

        # All discovery requests with the same service name are sent to the same server
        discovery_server, discovery_servers, discovery_version = \
            self._consistent_hash.get_node_nodes(service_name)
        if discovery_server != self._discovery_server:
            # request need sent to another discovery server
            status = discovery.Status(
                code=discovery.Code.REDIRECT, message=discovery_server)
            redirect_response = discovery.Response(
                status=status,
                discovery_version=discovery_version,
                discovery_servers=discovery_servers)
            return redirect_response

        new_version, servers = service.get_servers(client, version)
        if new_version > version:
            logging.info('client={} new_version={}, servers={}'.format(
                client, new_version, servers))

        status = discovery.Status(code=discovery.Code.OK)
        return discovery.Response(
            status=status,
            version=new_version,
            servers=servers,
            discovery_version=discovery_version,
            discovery_servers=discovery_servers)

    def start(self):
        # start db
        self._db.init()

        self._consistent_hash_thread = threading.Thread(
            target=self._consistent_hash_worker)
        self._consistent_hash_thread.daemon = True
        self._consistent_hash_thread.start()

        self._update_service_thread = threading.Thread(
            target=self._update_service_worker)
        self._update_service_thread.daemon = True
        self._update_service_thread.start()
