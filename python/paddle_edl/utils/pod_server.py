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


class PodServerServicer(master_pb2_grpc.MasterServicer):
    def __init__(self, master, reader_cls, file_list=None, capcity=3000):
        self._master = master
        # master.SubDataSetMeta
        self._sub_data_set = Queue()
        # {file_key:{rec_no: data}}
        self._data = {}
        # to control the cache size.
        self._data_queue = Queue(capcity)
        self._lock = Lock()
        self._file_list = file_list
        self._reader_cls = reader_cls

        assert type(reader_cls) == DataReader

        if self._master:
            self._t_get_sub_dataset = Thread(target=self._get_sub_dataset)
            self._t_get_sub_dataset.start()
        elif self._file_list:
            logger.info("init from list:{} ".format(self._file_list))
            arr = utils.get_file_list(self._file_list)
            for t in arr:
                logger.debug("readed:{} {}".format(t[0], t[1]))
                d = master_pb2.SubDataSetMeta()
                d.file_path = t[0]
                d.idx_in_list = t[1]
                self._sub_data_set.put(d)
        else:
            assert False, "You must set datasource"

        self._t_read_data = Thread(target=self._read_data)
        self._t_read_data.start()

    def GetSubDataSet(self, request, context):
        pass

    def Barrier(self, request, context):
        pass

    def ShutDown(self, request, context):
        logger.info("Enter into shutdown method")
        self._sub_data_set.put(None)
        self._t_read_data.join()
        return common_pb2.RPCRet()


class PodServer(object):
    def __init__(self):
        self._server = None
        self._port = None
        self_endpoint = None

    def start(self, job_env, pod, concurrency=20):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        master_pb2_grpc.add_PodServerServicer_to_server(
            PodServerServicer(
                master=master,
                data_set_reader=data_set_reader,
                capcity=cache_capcity,
                file_list=file_list),
            server)

        self._port = server.add_insecure_port('{}'.format(pod.addr))
        assert self._port > 0, "data server start on endpoint:{} error, selected port is {}".format(
            pod.addr, self._port)
        self._endpoint = "{}:{}".format(pod.addr, self._port)

        server.start()
        self._server = server
        pod.port = self._port

    def wait(self, timeout=None):
        if timeout is not None:
            self._server.stop(timeout)
            return
        self._server.wait_for_termination(timeout)
