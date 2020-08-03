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

from . import distill_discovery_pb2
from . import distill_discovery_pb2_grpc
import grpc
import logging

from concurrent import futures
from .balance_table import BalanceTable

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class DiscoveryServicer(distill_discovery_pb2_grpc.DiscoveryServiceServicer):
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

        return self._table.register_client(client, service_name, require_num)

    def HeartBeat(self, request, context):
        client = request.client
        version = request.version

        return self._table.get_servers(client, version)


def serve(server, worker_num, db_endpoints):
    discovery_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=worker_num))
    balance_table = BalanceTable(server, db_endpoints)

    distill_discovery_pb2_grpc.add_DiscoveryServiceServicer_to_server(
        DiscoveryServicer(balance_table), discovery_server)
    discovery_server.add_insecure_port(server)
    discovery_server.start()
    discovery_server.wait_for_termination()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Discovery server with balance')
    parser.add_argument(
        '--server',
        type=str,
        default='127.0.0.1:7001',
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
        default='127.0.0.1:2379',
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
        default='etcd',
        help='database type, only support etcd for now [default: %(default)s]')

    args = parser.parse_args()
    server = args.server
    worker_num = args.worker_num
    db_endpoints = args.db_endpoints.split(',')

    serve(server, worker_num, db_endpoints)
