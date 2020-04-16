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
import remote_executor_pb2
import remote_executor_pb2_grpc
import grpc
import sys
from utils.logger import logging
import queue


class RemoteExecutorServicer(object):
    def __init__(self, data_queue):
        self._data_queue = data_queue
        pass

    def SendData(self, request, context):
        ret = remote_executor_pb2.DataResponse()
        ret.err_code = 0

        self._data_queue.push(request)
        return ret


class RemoteExecutorServer(object):
    def __init__(self):
        self._data_queue = queue.Queue()
        self._paddle = None
        pass

    def start(self, endpoint, max_workers=1000, concurrency=100):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        data_server_pb2_grpc.add_DataServerServicer_to_server(
            RemoteExecutorServicer(self._data_queue), server)
        server.add_insecure_port('[::]:{}'.format(endpoint))
        server.start()
        server.wait_for_termination()

    def run(self):
        while True:
            data = self._data_queue.get()
            if data is None:
                logging.Fatal("Get None data from queue")
                time.sleep(3)
                continue
            self._paddle.execute(data)
        pass


if __name__ == "__main__":
    data_server = RemoteExecutorServer()
    data_server.start(endpoint=sys.argv[1])
