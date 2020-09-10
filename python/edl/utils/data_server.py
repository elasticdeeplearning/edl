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
from . import common_pb2 as common_pb
from . import data_server_pb2 as pb
from . import data_server_pb2_grpc as pb_grpc
import grpc
import sys
import os
import logging
from threading import Thread, Lock
from six.moves.queue import Queue
from .exceptions import *
import signal
import threading
import copy
from . import utils
from .utils import logger


class PodBatchData(object):
    def __init__(self, pod_id, data_server_endpoint, max_size=1000000):
        self._lock = threading.Lock()
        # batch_data_id=>BatchData
        self._cache = {}
        self._queue = Queue(max_size)
        self._pod_id = pod_id
        self._data_server_endpoint = data_server_endpoint
        self._max_size = max_size

    @property
    def size(self):
        with self._lock:
            return len(self._cache)

    @property
    def max_size():
        return self._max_size

    def pop(self):
        with self._lock:
            b = self._queue.pop()
            self._cache.pop(b.batch_data_id)

    def put(self, b):
        with self._lock:
            self._queue.put(b)
            self._cache[b.batch_data_id] = b


class ReaderPodData(object):
    def __init__(self, reader_name, file_list, pod_ids):
        self._reader_name = reader_name

        # pod_id => PodBatchData
        self._data = {}
        self._lock = Lock()

        # pod_id => [FileListElement]
        self._pod_file_list = {}
        self._file_list = file_list

        self._pod_ids = pod_ids

        self._init()

    def _init(self):
        for pod_id in pod_ids:
            self._data[pod_id] = PodBatchData()
            self._pod_file_list[pod_id] = []

        i = 0
        while i < len(self._file_list):
            for pod_id in pod_ids:
                m = pb.FileListElement()
                m.idx = i
                m.path = self._file_list[i]

                self._pod_file_list[pod_id].append(m)
                i += 1
                if i >= len(self._file_list):
                    break

    def get_pod_data(self, pod_id):
        with self._lock:
            if pod_id in self._data:
                return self._data[pod_id]
            return None

    def get_pod_file_list(self, pod_id):
        return self._pod_file_list[pod_id]


class DataServerServicer(pb_grpc.DataServerServicer):
    def __init__(self, trainer_env, reader_name, file_list, pod_ids):
        self._lock = threading.Lock()
        self._trainer_env = trainer_env
        self._file_list = file_list
        self._pod_ids = pod_ids

        # reader_name=>ReaderPodData
        self._reader_pod_data = ReaderPodData(reader_name, file_list, pod_ids)

    def _check_leader(self):
        if self._trainer_env.global_rank != 0:
            raise EdlNotLeaderError("This server is not Leader")

    def BalanceBatchData(self, request, context):
        pass

    def GetBatchData(self, request, context):
        pass

    def _check_file_list(self, file_list):
        for i in file_list:
            if self._file_list[i] != file_list[i]:
                raise EdlFileListNotMatchError("client:{} server:{}".format(
                    file_list, self._file_list))

    def _check_pod_id(self, pod_id):
        if pod_id not in self._pod_ids:
            raise EdlPodIDNotExistError("pod_id:{} not exist in {}".format(
                pod_id, self._pod_ids))

    def _check_reader_name(self, reader_name):
        if reader_name != self._reader_name:
            raise EdlReaderNameError("{} not equal {}".format(
                reader_name, self._reader_name))

    def GetFileList(self, request, context):
        res = FileListResponse()
        try:
            self._check_leader()
            self._check_file_list()
            self._check_pod_id()
            self._check_reader_name()

            pod_file_list = self._reader_pod_data.get_pod_file_list(
                request.pod_id)

            if m not in pod_file_list:
                res.file_list.append(m)

            return res
        except Exception as e:
            res.status = serialize_exception(e)
            return res


class DataServer(object):
    def __init__(self, trainer_env, reader_id, reader_name, data_checkpoint):
        self._server = None
        self._addr = None
        self._port = None
        self._endpoint = None

        self._trainer_env = trainer_env
        self._reader_id = reader_id
        self._reader_name = reader_name
        self._data_checkoint = data_checkpoint

    def start(self,
              addr,
              cache_capcity=1000,
              file_list=None,
              max_workers=100,
              concurrency=20):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        data_server_pb2_grpc.add_DataServerServicer_to_server(
            DataServerServicer(
                file_list=self._file_list, trainer_env=self._trainer_env),
            server)

        self._addr = addr
        self._port = server.add_insecure_port('{}:0'.format(addr))
        assert self._port > 0, "data server start on endpoint:{} error, selected port is {}".format(
            endpoint, self._port)
        self._endpoint = "{}:{}".format(self._addr, self._port)

        server.start()
        self._server = server

    @property
    def endpoint(self):
        return self._endpoint

    def wait(self, timeout=None):
        if timeout is not None:
            self._server.stop(timeout)
            return
        self._server.wait_for_termination(timeout)

    def shutdown(self):
        pass
