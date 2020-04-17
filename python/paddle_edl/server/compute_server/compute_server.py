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

from __future__ import print_function
from concurrent import futures
import compute_server_pb2
import compute_server_pb2_grpc
import grpc
import sys
from utils.logger import logging
import queue


class ComputeServicer(object):
    def __init__(self, data_queue):
        self._data_queue = data_queue
        pass

    def DoTasksByFilesMeta(self, request, context):
        ret = common_pb2.RPCRet()
        ret.err_code = 0
        ret.err_info = ""

        self._data_queue.push(request)
        return ret


class ComputeServer(object):
    def __init__(self):
        self._data_queue = queue.Queue()
        self._paddle = None
        self._server = None

    def init(self):
        # get program desc from job_server
        try:
            edl_env = Edlenv()
            program_desc = edl_env.get_program_desc()
            self._paddle.init(program_desc)
        except e:
            print("ComputeServer init error:{}".format(e))
            sys.exit(-1)

    def start(self, endpoint, max_workers=10, concurrency=10):
        # start sever
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        compute_server_pb2_grpc.add_ComputeServerServicer_to_server(
            RemoteExecutorServicer(self._data_queue), server)
        server.add_insecure_port('[::]:{}'.format(endpoint))
        server.start()
        self._server = server

    def run(self):
        # wait meta data to get real data.
        while True:
            request = self._data_queue.get()
            if request is None:
                logging.Fatal("Get None data from queue")
                time.sleep(3)
                continue
            self._paddle.execute(request)
        self._server.wait_for_termination()


if __name__ == "__main__":
    server = ComputeServer()
    server.init()
    server.start(endpoint=sys.argv[1])
    server.Run()
