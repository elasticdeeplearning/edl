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
import common_pb2
import master_pb2
import master_pb2_grpc
import grpc
import sys
import os
import logging
from threading import Thread, Lock
from Queue import Queue
from exception import *
from dataset import EdlDataSet, TxtDataSet
import utils
from utils import *
import signal
import threading


class DataServerServicer(data_server_pb2_grpc.DataServerServicer):
    def __init__(self, master, data_set_reader, file_list=None, capcity=1000):
        self._master = master
        # master.SubDataSetMeta
        self._sub_data_set = Queue()
        # {file_key:{rec_no: data}}
        self._data = {}
        # to control the cache size.
        self._data_queue = Queue(capcity)
        self._lock = Lock()
        self._file_list = file_list
        self._data_set_reader = data_set_reader

        assert isinstance(data_set_reader, EdlDataSet)

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

    def _get_sub_dataset(self):
        if self.master:
            channel = grpc.insecure_channel(self.master)
            stub = data_server_pb2_grpc.MasterStub(channel)
            while True:
                request = master_pb2.SubDataSetRequest()
                response = stub.GetSubDataSet(request)
                for file_data_set in response.files:
                    self._sub_data_set.put(file_data_set)

    def _get_file_key(self, idx, file_path):
        key = "idx:{}_path:{}".format(idx, file_path)
        return key

    def _read_data(self):
        while True:
            file_data_set = self._sub_data_set.get()
            if file_data_set is None:
                logger.info("terminated exit!")
                break

            rec_map = {}
            for one_range in file_data_set.filtered_records:
                for rec_no in range(one_range.begin, one_range.end + 1):
                    rec_map[rec.record_no] = one_range.status

            for rec_no, data in enumerate(
                    self._data_set_reader.reader(file_data_set.file_path)):
                if rec_no in rec_map and rec_map[
                        rec_no] == RecordStatus.PROCSSED:
                    continue

                logger.debug("read rec_no:{} data_len:{}".format(rec_no,
                                                                 len(data)))

                self._data_queue.put(1, block=True)
                key = self._get_file_key(file_data_set.idx_in_list,
                                         file_data_set.file_path)
                with self._lock:
                    if key not in self._data:
                        self._data[key] = {}
                    self._data[key][rec_no] = data

    def GetData(self, request, context):
        return self._get_data(request, context)

    def _get_data(self, request, context):
        try:
            response = data_server_pb2.DataResponse()
            files = data_server_pb2.Files()
            files_error = data_server_pb2.FilesError()

            for meta in request.metas:
                logger.debug("proc meta:{}".format(
                    utils.datameta_to_string(meta)))
                one_file = data_server_pb2.File()
                one_file.idx_in_list = meta.idx_in_list
                one_file.file_path = meta.file_path

                file_error = data_server_pb2.FileError()
                file_error.idx_in_list = meta.idx_in_list
                file_error.file_path = meta.file_path
                file_error.status = data_server_pb2.DataStatus.NOT_FOUND

                key = self._get_file_key(meta.idx_in_list, meta.file_path)
                logger.debug("getdata of file:{}".format(key))
                with self._lock:
                    if key not in self._data:
                        logger.error("file key:{} not found in cache".format(
                            key))
                        files_error.errors.append(file_error)
                        continue

                    record = data_server_pb2.Record()
                    record_error = data_server_pb2.RecordError()
                    record_error.status = data_server_pb2.DataStatus.NOT_FOUND

                    for one_range in meta.records:
                        for rec_no in range(one_range.begin,
                                            one_range.end + 1):
                            if rec_no not in self._data[key]:
                                record_error.record_no = rec_no
                                record_error.status = data_server_pb2.DataStatus.NOT_FOUND
                                file_error.errors.append(record_error)
                                logger.error(
                                    "file key:{} rec_no:{} not found in cache".
                                    format(key, rec_no))
                                continue

                            data = self._data[key][rec_no]

                            record.record_no = rec_no
                            record.data = data

                            one_file.records.append(record)

                    if len(file_error.errors) > 0:
                        files_error.errors.append(file_error)
                    files.files.append(one_file)

            if len(files_error.errors) > 0:
                response.errors.CopyFrom(files_error)
                return response

            response.files.CopyFrom(files)
            return response
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.fatal("context:{} {} {}".format(exc_type, fname,
                                                   exc_tb.tb_lineno))
            raise e

    def ClearDataCache(self, request, context):
        try:
            response = common_pb2.RPCRet()
            for meta in request.metas:
                file_key = self._get_file_key(meta.idx_in_list, meta.file_path)

                with self._lock:
                    if file_key not in self._data:
                        logger.error("file:{} not in cache:".format(file_key))
                        continue

                    recs = self._data[file_key]
                    for rec_no in recs.keys():
                        if rec_no not in recs:
                            logger.error("file:{} record_no:{} not in cache:".
                                         format(file_key, rec_no))
                            continue

                        recs.pop(rec_no)
                        self._data_queue.get(block=False)

            return response
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.fatal("context:{} {} {}".format(exc_type, fname,
                                                   exc_tb.tb_lineno))
            raise e

    def ShutDown(self, request, context):
        logger.info("Enter into shutdown method")
        self._sub_data_set.put(None)
        self._t_read_data.join()
        return common_pb2.RPCRet()


class DataServer(object):
    def __init__(self):
        self._server = None

    def start(self,
              endpoint,
              master,
              data_set_reader,
              cache_capcity=1000,
              file_list=None,
              max_workers=100,
              concurrency=10):
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
        server.add_insecure_port('{}'.format(endpoint))
        server.start()

        self._server = server

    def wait(self, timeout=None):
        if timeout is not None:
            self._server.stop(timeout)
            return
        self._server.wait_for_termination(timeout)


if __name__ == '__main__':

    logger = get_logger(10)

    endpoint = "0.0.0.0:6700"
    data_server = DataServer()
    data_server.start(
        endpoint=endpoint,
        data_set_reader=TxtDataSet(),
        file_list="./test_file_list.txt",
        master=None)
    data_server.wait(2)
