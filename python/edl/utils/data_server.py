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
from . import data_server_pb2
from . import data_server_pb2_grpc
from . import common_pb2
from . import master_pb2
from . import master_pb2_grpc
import grpc
import sys
import os
import logging
from threading import Thread, Lock
from six.moves.queue import Queue
from .exception import *
from .dataset import DataReader
import signal
import threading
import copy
import paddle_edl.utils.utils as utils
from .utils import logger


class DataServerServicer(data_server_pb2_grpc.DataServerServicer):
    def __init__(self, file_list, world_rank, self_rank):
        self._file_list = file_list
        self._lock = Threading.Lock()

        # rank=>file_list
        self._trainer_file_list = {}
        self._initital()

        # rank => batch_data_meta
        self._trainer_batch_data_meta = {}

        self._rank = self_rank

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

                res.status = pb.Status()
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
    def __init__(self):
        self._server = None
        self._port = None
        self_endpoint = None

    def start(self,
              master,
              addr,
              port,
              data_set_reader,
              job_env,
              pod_id,
              rank_of_pod,
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
                master=master,
                data_set_reader=data_set_reader,
                capcity=cache_capcity,
                file_list=file_list),
            server)

        endpoint = "{}:{}".format(addr, port)
        self._port = server.add_insecure_port('{}'.format(endpoint))
        assert self._port > 0, "data server start on endpoint:{} error, selected port is {}".format(
            endpoint, self._port)
        self._endpoint = "{}:{}".format(addr, self._port)

        server.start()
        self._server = server

        self._register = DataServerRegister(
            job_env.etcd_endoints,
            job_env.job_id,
            affinity_pod_id=affinity_pod_id,
            affinity_rank_of_pod=affinity_rank_of_pod,
            endpoint=self._endpoint)

    @property
    def port(self):
        return self._port

    def wait(self, timeout=None):
        if timeout is not None:
            self._server.stop(timeout)
            return
        self._server.wait_for_termination(timeout)

    def shutdown(self):
        pass
