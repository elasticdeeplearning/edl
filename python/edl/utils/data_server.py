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

import grpc
import threading
from concurrent import futures
from six.moves.queue import Queue
from threading import Lock
from random import shuffle
import collections

from . import data_server_pb2 as pb
from . import data_server_pb2_grpc as pb_grpc
from . import error_utils
from . import exceptions


class PodData(object):
    """
    Manage pod's data:
    batch_data_ids, file_list, data_server_endpoint
    """

    def __init__(self, pod_id, data_server_endpoint, max_size=1000000):
        # batch_data_ids
        self._pod_id = pod_id
        self._data_server_endpoint = data_server_endpoint
        # total ids for filter
        self._batch_data_ids = set()

        self._queue = collections.deque(max_size)
        self._pod_file_list = []
        self._max_size = max_size
        self._reach_data_end = False

    def append_file_list_element(self, element):
        self._file_list.append(element)

    @property
    def reach_data_end(self):
        return self._reach_data_end

    @reach_data_end.setter
    def reach_data_end(self, r):
        self._reach_data_end = r

    @property
    def size(self):
        return len(self._batch_data)

    @property
    def max_size():
        return self._max_size

    def pop(self, num):
        a = []
        while not self._queue.empty():
            if (num > 0 and len(a) < num) or num <= 0:
                batch_data_id = self._queue.popleft()
                a.append(batch_data_id)
            else:
                break

        return a

    def put(self, batch_data_ids):
        for batch_data_id in batch_data_ids:
            if batch_data_id in self._batch_data_ids:
                continue
            self._queue.append(batch_data_id)
            self._batch_data_ids.add(batch_data_id)


class PodsData(object):
    """
    Reader's pods data
    pod_id=>PodData
    """

    def __init__(self, reader_name, file_list, pod_ids):
        self._reader_name = reader_name

        # pod_id => PodData
        self._pods_data = {}
        # pod_id => BalanceBatchData
        self._balance_batch_data = {}
        self._barrier_ids = set()
        self._reach_data_end_ids = set()
        self._lock = Lock()

        self._file_list = file_list
        self._pod_ids = set(pod_ids)

        self._init()
        self._total = 0

    def _init(self):
        for pod_id in pod_ids:
            self._data[pod_id] = PodData()
            self._pod_file_list[pod_id] = []
            self._balance_batch_data[pod_id] = [
            ]  # array of pb.BalanceBatchData()

        i = 0
        while i < len(self._file_list):
            for pod_id in pod_ids:
                m = pb.FileListElement()
                m.idx = i
                m.path = self._file_list[i]

                self._pods_data[pod_id].append_file_list_element(m)
                i += 1
                if i >= len(self._file_list):
                    break

    def get_pod_file_list(self, pod_id):
        return self._pod_file_list[pod_id]

    def set_data_end(self, pod_id):
        with self._lock:
            pod_data = self._pods_data[pod_id]
            pod_data.reach_data_end()
            self._reach_data_end_ids.add(pod_id)

    def _get_batch_data_id_from_others(self, avg_num, need_num):
        ret = []
        for pod_id in self._pod_ids:
            src = self._pods_data[pod_id]
            if src.size() < avg_num:
                continue

            dst = pb.BalanceBatchData()
            dst.reader_name = self._reader_name
            dst.producer_pod_id = src._pod_id
            dst.consumer_pod_id = None
            dst.data_server_endpoint = src._data_server_endpoint

            pop_num = (src.size() - avg_num)
            ids = src.pop(pop_num)
            if len(ids) <= 0:
                continue

            dst.extend(ids)
            ret.append(dst)
            need_num -= len(ids)

            if need_num <= 0:
                break

        return ret

    def put(self, pod_id, batch_data_ids):
        total = 0
        with self._lock:
            pod_data = self._pods_data[pod_id]
            pod_data.put(batch_data_ids)

            for _, pod_data in self._pods_data:
                total += pod_data.size()

            self._barrier_ids.add(pod_id)
            if self._barrier_ids + self._reach_data_end_ids != self._pod_ids:
                return

            avg_num = self._total / len(self._pods_ids)
            if avg_num < 1:
                return

            # get batch_data_ids from pods_data to balance_batch_data
            for pod_id in self._pod_ids:
                src = self._pods_data[pod_id]

                dst = pb.BalanceBatchData()
                dst.reader_name = self._reader_name
                dst.producer_pod_id = src._pod_id
                dst.consumer_pod_id = None
                dst.data_server_endpoint = src._data_server_endpoint

                ids = src.pop(num=avg_num)
                if len(ids) >= avg_num:
                    dst.batch_data_ids.extend(ids)
                    self._balance_batch_data.append(dst)
                else:
                    need_num = avg_num - len(ids)
                    ret = self._get_batch_data_id_from_others(avg_num,
                                                              need_num)
                    if len(ret) <= 0:
                        continue
                    self._balance_batch_data.extend(ret)

    def _is_all_reach_data_end(self):
        for _, pod_data in six.iteritem(self._pods_data):
            if not pod_data.reach_data_end:
                return False

        return True

    # FIXME(gongwb): avoid global lock of all pods
    @error_utils.handle_errors_until_timeout
    def pop(self, pod_id, ret, timeout=60):
        with self._lock:
            balanced_data = self._balance_batch_data[pod_id]

            if len(balanced_data) > 0:
                return balanced_data

            if self._is_all_reach_data_end():
                return None

            raise exceptions.EdlDataGenerateError("wait to generate more data")


class DataServerServicer(pb_grpc.DataServerServicer):
    def __init__(self, trainer_env, reader_name, file_list, pod_ids,
                 local_data):
        self._lock = threading.Lock()
        self._trainer_env = trainer_env
        self._file_list = file_list
        self._pod_ids = pod_ids
        self._local_data = local_data

        # reader_name=>PodData
        self._pods_data = PodsData(reader_name, file_list, pod_ids)

    def _check_leader(self):
        if self._trainer_env.global_rank != 0:
            raise exceptions.EdlNotLeaderError("This server is not Leader")

    # only leader can do this
    def ReportBatchDataIds(self, request, context):
        res = common_pb.Empty()
        try:
            self._check_leader()
            self._check_pod_id(request.pod_id)
            self._check_reader_name(request.reader_name)

            if len(request.batch_data_ids) > 0:
                self._pods_data.put(request.batch_data_ids)
            else:
                self._pods_data.set_data_end(request.pod_id)

            return res
        except Exception as e:
            res.status = exceptions.serialize_exception(e)
            return res

    # only leader can do this
    def GetBatchDataIds(self, request, context):
        res = BatchDataResponse()
        try:
            self._check_leader()
            self._check_pod_id(request.pod_id)
            self._check_reader_name(request.reader_name)

            self._pods_data.pop(request.pod_id, res.data)
            return res
        except Exception as e:
            res.status = exceptions.serialize_exception(e)
            return res

    def GetBatchData(self, request, context):
        res = BatchDataResponse()
        try:
            datas = self._local_data.get_local_batch_data(request)
            for data in datas:
                b = copy.copy(data)
                res.datas.append(b)
        except Exception as e:
            res.status = exceptions.serialize_exception(e)
        return res

    def _check_file_list(self, file_list):
        for i in file_list:
            if self._file_list[i] != file_list[i]:
                raise exceptions.EdlFileListNotMatchError(
                    "client:{} server:{}".format(file_list, self._file_list))

    def _check_pod_id(self, pod_id):
        if pod_id not in self._pod_ids:
            raise exceptions.EdlPodIDNotExistError(
                "pod_id:{} not exist in {}".format(pod_id, self._pod_ids))

    def _check_reader_name(self, reader_name):
        if reader_name != self._reader_name:
            raise exceptions.EdlReaderNameError("{} not equal {}".format(
                reader_name, self._reader_name))

    # only leader can do this
    def GetFileList(self, request, context):
        res = FileListResponse()
        try:
            self._check_leader()
            self._check_file_list(request.file_list)
            self._check_pod_id(request.pod_id)
            self._check_reader_name(request.reader_name)

            pod_file_list = self._pods_data.get_pod_file_list(request.pod_id)

            if m not in pod_file_list:
                res.file_list.append(m)

            return res
        except Exception as e:
            res.status = exceptions.serialize_exception(e)
            return res


class DataServer(object):
    def __init__(self, trainer_env, reader_name, file_list, local_data):
        self._server = None
        self._addr = None
        self._port = None
        self._endpoint = None

        self._trainer_env = trainer_env
        self._reader_name = reader_name
        self._file_list = file_list
        self._local_data = local_data

    def start(self, addr, cache_capcity=1000, max_workers=100, concurrency=20):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        data_server_pb2_grpc.add_DataServerServicer_to_server(
            DataServerServicer(
                file_list=self._file_list,
                trainer_env=self._trainer_env,
                local_data=local_data),
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
