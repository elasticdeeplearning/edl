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

import redis


class RedisStore(object):
    def __init__(self, ip='127.0.0.1', port=6379, passwd=None):
        self._ip = ip
        self._port = port
        self._passwd = passwd
        self._redis = redis.Redis(
            host=ip, port=port, password=passwd, decode_responses=True)
        print("connected to redis ip:{} port:{}".format(ip, port))

    def get_service(self, service_name):
        servers = []
        for key in self._redis.scan_iter('/service/{}/nodes/*'.format(
                service_name)):
            servers.append(self._redis.hgetall(key))
        return servers

    def remove_service(self, service_name):
        for key in self._redis.scan_iter('/service/{}/*'.format(service_name)):
            self._redis.delete(key)

    def set_server(self, service_name, server, info, ttl=6):
        server_info = {'server': server, 'info': info}
        key = '/service/{}/nodes/{}'.format(service_name, server)
        self._redis.hmset(key, server_info)
        self._redis.expire(key, ttl)

    def remove_server(self, service_name, server):
        self._redis.delete('/service/{}/nodes/{}'.format(service_name, server))

    def refresh(self, service_name, server, info=None, ttl=6):
        if info is not None:
            self.set_server(self, service_name, server, info, ttl)
            return True
        key = '/service/{}/nodes/{}'.format(service_name, server)
        time = self._redis.ttl(key)
        if time < 0:
            return False
        self._redis.expire(key, ttl)
        return True

    def get_client(self, client):
        # Todo
        pass

    def set_client(self, client, service_name):
        # Todo
        pass


if __name__ == '__main__':
    service_name = 'TestService'
    store = RedisStore('127.0.0.1', 6379)
    print(store.get_service(service_name))
    store.set_server(service_name, '127.0.0.1:5454', '{cpu: 10%, gpu: 20%}')
    print(store.get_service(service_name))
