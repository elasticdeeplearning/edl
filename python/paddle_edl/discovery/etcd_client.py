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


class NoValidEndpoint(Exception):
    pass


def _handle_errors(f):
    def handler(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except exceptions.Etcd3Exception as e:
            args[0]._connect(args[0]._endpoints)

        return f(*args, **kwargs)

    return functools.wraps(f)(handler)


class EtcdClient(object):
    def __init__(self, endpoints=['127.0.0.1:2379'], passwd=None):
        self._endpoints = set(endpoints)
        self._passwd = passwd
        self._etcd = None
        self._leases = {}

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
        d = '/service/{}/nodes/'.format(service_name)
        for value, meta in self._etcd.get_prefix(d):
            servers.append([
                self.get_server_name_from_all_path(meta.key, service_name),
                value
            ])
        return servers

    def watch_service(self, service_name, call_back):
        d = '/service/{}/'.format(service_name)
        return self._etcd.add_watch_prefix_callback(d, call_back)

    def cancel_watch(self, watch_id):
        return self._etcd.cancel_watch(watch_id)

    @_handle_errors
    def remove_service(self, service_name):
        d = '/service/{}/'.format(service_name)
        servers = self.get_service(service_name)
        for s in servers:
            self.remove_server(service_name, s[0])
        self._etcd.delete_prefix(d)

    @_handle_errors
    def _get_lease(self, key, ttl=6):
        if key not in self._leases:
            lease = self._etcd.lease(ttl)
            self._leases[key] = lease

        return self._leases[key]

    @_handle_errors
    def set_server(self, service_name, server, info, ttl=6):
        key = '/service/{}/nodes/{}'.format(service_name, server)
        lease = self._get_lease(key, ttl)
        self._etcd.put(key=key, value=info, lease=lease)

    @_handle_errors
    def remove_server(self, service_name, server):
        key = '/service/{}/nodes/{}'.format(service_name, server)
        self._etcd.delete(key)
        if key in self._leases:
            lease = self._leases[key]

            self._etcd.revoke_lease(lease.id)
            self._leases.pop(key)

    @_handle_errors
    def refresh(self, service_name, server, info=None, ttl=6):
        if info is not None:
            self.set_server(service_name, server, info, ttl)
            return

        key = '/service/{}/nodes/{}'.format(service_name, server)
        lease = self._get_lease(key, ttl)

        self._etcd.refresh_lease(lease)
        return

    @staticmethod
    def get_server_name_from_all_path(path, service_name):
        d = '/service/{}/nodes/'.format(service_name)
        return path[len(d):]
