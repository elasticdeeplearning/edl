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
import threading
import time
import json
import uuid
import copy
import traceback
import six
import multiprocessing
from __future__ import print_function

from .utils import logger
from .pod import Pod
from ..discovery.etcd_client import EtcdClient

import etcd3
from .global_vars import *
from .cluster import Cluster
from .exceptions import EdlGenerateClusterError, EdlTableError
from .etcd_db import get_global_etcd

from .utils.edl_env import TrainerEnv
from .utils import handle_timeout_errors
from .unique_name import generator
from .data_server_client import DataServerClient
from . import data_server_pb2 as pb


class DataGenerator(ProcessWrapper):
    def __init__(self, reader_leader_endpoint, reader_name, pod_id,
                 total_file_list, splitter_cls, out_queue):
        super(DataGenerator, self).__init__()

        self._batch_data_id = 0

        self._leader_endpoint = leader_endpoint
        self._pod_id = pod_id
        self._reader_name = reader_name

        self._file_list = total_file_list
        self._splitter_cls = splitter_cls
        self._data_queue = out_queue

    def _get_file_list(self, timeout=60):
        client = DataServerClient()
        return client.get_file_list(
            leader_endpoint=self._leader_endpoint,
            reader_name=self._reader_name,
            pod_id=self._pod_id,
            file_list=self._file_list)

    def _generate_batch_data(self):
        self._batch_data_id += 1
        b = pb.BatchData()
        b.batch_data_id = self._batch_data_id
        b.data = None

        return b

    def _read_batch_data(self):
        while True:
            b = self._generate_batch_data()
            for m in _get_file_list():
                if self._stop.set():
                    break

                assert self._file_list[m.idx] == m.path
                fields = self._splitter_cls(m.path)
                for i in o:
                    assert fields[0] == m.idx
                    rec = pb.Record()
                    rec.record_no = fields[0]
                    for field in fields[1:]:
                        rec.field_data.append(field)

                    if len(b.records) >= self._batch_size:
                        self._data_queue.put(b)
                        b = self._generate_batch_data()

            if len(b.records) > 0:
                self._data_queue.put(b)

            self._data_queue.put(None)

    def _worker_func(self):
        try:
            self._read_batch_data()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)


class DataAccesser(object):
    def __init__(self, leader, reader_name, trainer_env, input_queue,
                 out_queue, queue_size):
        super(DataGenerator, self).__init__()

        self._reader_leader = leader
        self._reader_name = reader_name
        self._trainer_env = trainer_env

        # BatchData
        self._input_queue = input_queue
        self._out_queue = out_queue
        # batch_data_id => BatchData
        self._cache = {}

        # BatchDataRequest
        self._req_queue = threading.Queue(queue_size)

        self._data_server = DataServer(self)
        self._data_server.start()

        self._stop = threading.Event()
        self._t_accesser = threading.Thread(target=self.access)
        self._t_generater = threading.Thread(target=self.generate)

        self._client = DataServerClient()

    def start(self):
        self._client.connect(self._reader_leader.endpoint)
        self._t_accesser.start()
        self._t_generater.start()

    def _access(self, report_size=10):
        a = []
        while not self._stop.set():
            while len(a) < report_size:
                b = self._input_queue.pop()
                if b is None:
                    logger.info("data read to end!")
                    break
                a.append(b.batch_data_id)
                with self._lock:
                    self._cache[b.batch_data_id] = b

            ret = self._client.balance_batch_data(
                reader_leader_endpoint=self._reader_leader.endpoint,
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
                endpoint=self._reader_leader.endpoint,
                batch_data_ids=a)

            self._req_queue.put(ret)
            a = []

        while not self._stop.set():
            ret = self._client.balance_batch_data(
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
                endpoint=self._reader_leader.endpoint,
                batch_data_ids=a)

            self._req_queue.put(ret)

        # data end
        self._req_queue.put(None)

    def _get_batch_data(self, req):
        if self._trainer_env.pod_id != req.producer_pod_id:
            return self._client.get_batch_data(req)

        return self._get_local_batch_data(req)

    def get_local_batch_data(self, req):
        ret = []
        for batch_data_id in req.data.batch_data_ids:
            with self._lock:
                ret.append(self._cache[batch_data_id])

        return ret

    def _generate(self):
        while not self._stop.set():
            meta = self._req_queue.pop()
            if meta is None:
                break

            ret = self._get_batch_data(meta)
            for b in ret:
                self._out_queue.put(b)

        self._out_queue.put(None)

    def access(self):
        try:
            self._access()
        except Exception as e:
            sys.exit(1)

    def generate(self):
        try:
            self._generate()
        except Exception as e:
            sys.exit(1)


def access_data(self, reader_leader, reader_name, trainer_env, input_queue,
                out_queue, cache_size):
    try:
        a = DataAccesser(reader_leader, reader_name, trainer_env, input_queue,
                         out_queue, queue_size)
        a.start()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


class DistributeReader(object):
    def __init__(self,
                 file_list,
                 file_splitter_cls,
                 batch_size,
                 cache_size=100):
        self._file_list = file_list
        assert isinstance(self._file_list, list), "file_list must be a list"

        self._name = generator("_dist_reader_")

        self._cls = file_splitter_cls
        self._batch_size = batch_size
        assert self._batch_size > 0, "batch size must > 0"
        self._cache_size = cache_size

        # connections to data servers
        self._trainer_env = TrainerEnv()

        self._db = get_global_etcd(trainer_env.endpoints, trainer_env.job_id)
        self._wait_record_to_dist_reader_table()
        self._wait_dist_reader_leader()

        self._generater_out_queue = multiprocessing.Queue(self._cache_size)
        self._accesser_out_queue = multiprocessing.Queue(self._cache_size)

        self._generater = None
        self._accesser = None

    @handle_errors_until_timeout
    def _wait_dist_reader_leader(self, timeout=120):
        self._reader_leader = self._db.get_dist_reader_leader()

    @handle_errors_until_timeout
    def _record_to_dist_reader_table(self, timeout=120):
        self._db.record_to_dist_reader_table(self._trainer_env.etcd_endpoint,
                                             self._name,
                                             self._trainer_env.pod_id)

    def stop(self):
        if self._generater:
            self._generater.stop()
            self._generater = None

        if self._accesser:
            self._accesser.terminate()
            self._accesser.join()
            self._accesser = None

    def __exit__(self):
        self.stop()

    def __iter__(self):
        self._generater = DataGenerator()
        self._generator.start()

        self._accesser = multiprocessing.Process(
            access_data,
            args=(self._reader_leader, self._name, self._trainer_env,
                  self._generater_out_queue, self._accesser_out_queue,
                  cache_size))
        while True:
            b = self._accesser_out_queue.pop()
            if b is None:
                logger.debug("{} reach data end".format(self._name))
                break
            yield b["meta"], b["data"]
