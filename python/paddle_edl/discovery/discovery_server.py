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

import discovery_pb2
import discovery_pb2_grpc
import grpc
import logging

from concurrent import futures
from balance_table import BalanceTable

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class DiscoveryServicer(discovery_pb2_grpc.DiscoveryServiceServicer):
    def __init__(self, table):
        self._table = table
        self._table.start()

    def Register(self, request, context):
        client = request.client
        service_name = request.service_name
        require_num = request.require_num
        token = request.token
        logging.info('client={}, service_name={}, require_num={} token={}'.
                     format(client, service_name, require_num, token))

        # TODO, return code
        self._table.register_client(client, service_name, require_num)

        return discovery_pb2.Response(msg='success', version=0, servers=[])

    def HeartBeat(self, request, context):
        client = request.client
        version = request.version

        ret = self._table.get_servers(client, version)
        if ret is None:
            # not register, or timeout
            return discovery_pb2.Response(
                msg='failed', version=None, servers=None)

        new_version, servers = ret
        logging.debug('client={} new_version={}, servers={}'.format(
            client, new_version, servers))
        return discovery_pb2.Response(
            msg='success', version=new_version, servers=servers)

    def UnRegister(self, request, context):
        logging.info('request={}, context={}'.format(request, context))
        return discovery_pb2.Response(msg='a', version=0, servers=['b', 'c'])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    table = BalanceTable(['127.0.0.1:2379'])
    discovery_pb2_grpc.add_DiscoveryServiceServicer_to_server(
        DiscoveryServicer(table), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
