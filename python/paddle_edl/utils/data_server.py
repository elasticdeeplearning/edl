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
import data_server_pb2
import data_server_pb2_grpc
import grpc
import sys
import logging
import threading
import DistributeReader
from exception import *
from dataset import EdlDataSet


class DataServerServicer(object):
    def __init__(self, master, data_set, capcity=1000):
        self._master = master
        self._sub_data_set = Queue()
        self._data = {}
        self._data_queue = Queue(capcity)
        self._lock = threading.Lock

        assert isinstance(data_set, EdlDataSet)

    def _get_sub_dataset(self):
        pass

    def _read_data(self):
        if self._sub_data_set.empty():
            try:
                sub_data_set = self._get_sub_dataset(self)

            except DataSetEndException as e:
                # return to client epoch reaches end.
                pass

        while not self._sub_data_set.empty():
            file_data_set = self._sub_data_set.pop()
            rec_map = {}
            for rec in file_data_set.record:
                rec_map[rec.record_no] = rec.status
            for rec_no, data in self._data_set.read(file_data_set.file_path):
                if rec_map[rec_no] == RecordStatus.PROCSSED:
                    continue

                key = "idx:{}_recno:{}".format(file_data_set.idx_in_list,
                                               rec_no)
                self._data_queue.put(1, block=True)
                with self._lock:
                    self._data[key] = data

    def GetData(self, request, context):
        for rec in request.record_no:
            key = "idx:{}_recno:{}".format(file_data_set.idx_in_list, rec_no)
            return self._data[key]

    def ClearDataCache(self, request, context):
        for rec in request.record_no:
            key = "idx:{}_recno:{}".format(file_data_set.idx_in_list, rec_no)
            with self._lock():
                self._data.pop(key)
            self._data_queue.pop()


class DataServer(object):
    def __init__(self):
        pass

    def start(self, max_workers=1000, concurrency=100, endpoint=""):
        if endpoint == "":
            logging.error("You should specify endpoint in start function")
            return

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        data_server_pb2_grpc.add_DataServerServicer_to_server(
            DataServerServicer(), server)
        server.add_insecure_port('[::]:{}'.format(endpoint))
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    data_server = DataServer()
    data_server.start(endpoint=sys.argv[1])
