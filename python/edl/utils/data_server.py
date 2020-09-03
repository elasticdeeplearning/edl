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


class DataServerServicer(pb_grpc.DataServerServicer):
    def __init__(self, file_list, trainer_env, data_checkpoint):
        self._file_list = file_list
        self._trainer_env = trainer_env
        self._lock = Threading.Lock()
        self._checkpoint = data_checkpoint

    def to_json(self):
        pass

    def from_json(self):
        pass

    def _initital(self):
        for i, f in enumerate(self._file_list):
            if i not in self._dispatched:
                self._trainer_file_list[i] = []
            self._trainer_file_list[i].append(f)

    def _check_leader(self, self_rank):
        if self._rank != 0:
            raise EdlNotLeaderError("This server is not Leader")

    def GetBatchData(self, request, context):
        """
        try to get data from loader's queue and return.
        """
        pass

    def GetBatchDataMeta(self, request, context):
        pass

    def GetFileList(self, request, context):
        res = FileListResponse()
        try:
            with self._lock:
                if i not in self._trainer_file_list[i]:
                    raise EdlFileListNotFoundError(
                        "can't get filelist of rank:{} id:{}".format(
                            request.rank, request.id))

                res.status = common_pb.Status()
                res.metas
                for f in self._trainer_file_list[i]:
                    meta = pb.Meta()
                    res.metas.append(meta)
            return res
        except Exception as e:
            res.status = serialize_exception(e)
            return res

    def SaveCheckpoint(self, request, context):
        pass

    def ShutDown(self, request, context):
        pass


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
                file_list=file_list,
                trainer_env=trainer_env,
                data_checkpoint=data_checkpoint,
                capcity=cache_capcity,
                file_list=file_list),
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
