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

import etcd3 as etcd
import etcd3.exceptions as exceptions
import functools
import json
import random
import logging
import time


class NoValidEndpoint(Exception):
    pass


class ServerMeta(object):
    def __init__(self, server, info, mod_revision, revision):
        self.server = server
        self.info = info
        self.mod_revision = mod_revision  # key mod revision
        self.revision = revision  # etcd global revision

    def __str__(self):
        return 'server={}, info={}, mod_revision={}, revision={}'.\
            format(self.server, self.info, self.mod_revision, self.revision)


def _handle_errors(f):
    def handler(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except exceptions.Etcd3Exception as e:
            args[0]._connect(args[0]._endpoints)

        return f(*args, **kwargs)

    return functools.wraps(f)(handler)


class EtcdClient(object):
    def __init__(self,
                 endpoints=['127.0.0.1:2379'],
                 passwd=None,
                 root="service"):
        self._endpoints = set(endpoints)
        self._passwd = passwd
        self._etcd = None
        self._leases = {}
        self._root = root

    def _endpoint_to_ip_port(self, endpoint):
        a = endpoint.split(":")
        return a[0], a[1]

    def _connect(self, endpoints):
        conn = None
        ep_lst = list(endpoints)
        random.shuffle(ep_lst)
        for ep in ep_lst:
            try:
                ip, port = self._endpoint_to_ip_port(ep)
                conn = etcd.client(host=ip, port=port)
            except Exception as e:
                print(e)
                continue

            self._etcd = conn
            return

        if self._etcd == None:
            raise NoValidEndpoint()

    def init(self):
        return self._connect(self._endpoints)

    @_handle_errors
    def get_service(self, service_name):
        servers = []
        d = '/{}/{}/nodes/'.format(self._root, service_name)
        for value, meta in self._etcd.get_prefix(d):
            servers.append(
                ServerMeta(
                    self.get_server_name_from_full_path(
                        meta.key, service_name), value, meta.mod_revision,
                    meta.response_header.revision))
        return servers

    @_handle_errors
    def get_service_with_revision(self, service_name):
        servers = []
        d = '/{}/{}/nodes/'.format(self._root, service_name)

        key_prefix = d
        range_response = self._etcd.get_prefix_response(key_prefix)
        for kv in range_response.kvs:
            servers.append(
                ServerMeta(
                    self.get_server_name_from_full_path(kv.key, service_name),
                    kv.value, kv.mod_revision, range_response.header.revision))

        return servers, range_response.header.revision

    def watch_service(self, service_name, call_back, **kwargs):
        # call_back(add_servers, rm_servers)
        #   add_servers: list(ServerMeta)
        #   rm_servers: list(ServerMeta)
        # start_revision=, watch start from revision
        def services_change(response):
            add_servers = dict()
            rm_servers = dict()
            for event in response.events:
                key = self.get_server_name_from_full_path(event.key,
                                                          service_name)
                server = ServerMeta(key, event.value, event.mod_revision,
                                    response.header.revision)

                if isinstance(event, etcd.events.PutEvent):
                    if key in rm_servers:
                        # after rm key, add again, need remove key from rm set
                        rm_servers.pop(key)
                    add_servers[key] = server
                elif isinstance(event, etcd.events.DeleteEvent):
                    if key in add_servers:
                        # after add key, rm again, need remove key from add dict
                        add_servers.pop(key)
                    rm_servers[key] = server
                else:
                    raise TypeError('ectd event type is not put or delete!')

            if len(add_servers) == 0 and len(rm_servers) == 0:
                return

            call_back(add_servers.values(), rm_servers.values())

        d = '/{}/{}/'.format(self._root, service_name)
        return self._etcd.add_watch_prefix_callback(d, services_change,
                                                    **kwargs)

    def cancel_watch(self, watch_id):
        return self._etcd.cancel_watch(watch_id)

    @_handle_errors
    def remove_service(self, service_name):
        d = '/{}/{}/'.format(self._root, service_name)
        servers = self.get_service(service_name)
        for s in servers:
            self.remove_server(service_name, s.server)
        self._etcd.delete_prefix(d)

    @_handle_errors
    def _get_lease(self, key, ttl=10):
        if key not in self._leases:
            lease = self._etcd.lease(ttl)
            self._leases[key] = lease

        return self._leases[key]

    @_handle_errors
    def set_server_not_exists(self,
                              service_name,
                              server,
                              info,
                              ttl=10,
                              timeout=20):
        """
        :returns: state of transaction, ``True`` if the put was successful,
                  ``False`` otherwise
        """
        key = '/{}/{}/nodes/{}'.format(self._root, service_name, server)
        lease = self._get_lease(key, ttl)
        begin = time.time()
        while True:
            if self._etcd.put_if_not_exists(key=key, value=info, lease=lease):
                return True

            # refresh lease
            for r in self._etcd.refresh_lease(lease.id):
                pass

            if (time.time() - begin > timeout):
                break

            time.sleep(1)
        return False

    @_handle_errors
    def _set_server(self, service_name, server, info, ttl=10):
        key = '/{}/{}/nodes/{}'.format(self._root, service_name, server)
        lease = self._get_lease(key, ttl)
        return self._etcd.put(key=key, value=info, lease=lease)

    @_handle_errors
    def _get_server(self, service_name, server):
        # for debug
        key = '/{}/{}/nodes/{}'.format(self._root, service_name, server)
        value, meta = self._etcd.get(key=key)
        return value, meta.key, meta.version, meta.create_revision, meta.mod_revision

    @_handle_errors
    def remove_server(self, service_name, server):
        key = '/{}/{}/nodes/{}'.format(self._root, service_name, server)
        self._etcd.delete(key)
        if key in self._leases:
            lease = self._leases[key]

            self._etcd.revoke_lease(lease.id)
            self._leases.pop(key)

    @_handle_errors
    def refresh(self, service_name, server, info=None, ttl=10):
        if info is not None:
            self._set_server(service_name, server, info, ttl)
            return

        key = '/{}/{}/nodes/{}'.format(self._root, service_name, server)
        lease = self._get_lease(key, ttl)

        for r in self._etcd.refresh_lease(lease.id):
            pass
        return

    def get_server_name_from_full_path(self, path, service_name):
        d = '/{}/{}/nodes/'.format(self._root, service_name)
        return path[len(d):]

    @_handle_errors
    def lock(self, service_name, server, ttl=10):
        key = '/{}/{}/nodes/{}'.format(self._root, service_name, server)
        return self._etcd.lock(key, ttl=ttl)

    @_handle_errors
    def get_key(self, key):
        return self._etcd.get(key)
