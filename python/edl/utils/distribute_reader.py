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


class BatchData(object):
    pass


class DataGenerator(ProcessWrapper):
    def __init__(self, leader_endpoint, reader_name, pod_id, total_file_list,
                 splitter_cls, data_queue):
        super(DataGenerator, self).__init__()

        self._batch_data_id = 0

        self._leader_endpoint = leader_endpoint
        self._pod_id = pod_id
        self._reader_name = reader_name

        self._file_list = total_file_list
        self._splitter_cls = splitter_cls
        self._data_queue = data_queue

    @handle_errors_until_timeout
    def _get_file_list(self, timeout=120):
        client = DataServerClient()
        return client.get_file_list(self.leader_endpoint, self._reader_name,
                                    self._pod_id)

    def _generate_batch_data(self):
        self._batch_data_id += 1
        return {
            "id:": self._batch_data_id,
            "data": None,  # {idx=>(record_no,(field_data...))... }
        }

    def _read_batch_data(self):
        while True:
            b = self._generate_batch_data()
            for m in _get_file_list():
                if self._stop.set():
                    break

                assert self._file_list[m.idx] == m.path
                fields = self._cls(m.path)
                for i in o:
                    assert fields[0] == m.idx
                    b["meta"]["idx"] = fields[0]
                    b["data"].append((fields[1:]))

                    if len(b['data']) >= self._batch_size:
                        self._data_queue.put(b)
                        b = self._generate_batch_data()

            if len(b["data"]) > 0:
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
                 out_queue):
        super(DataGenerator, self).__init__()

        self._reader_name = reader_name
        self._leader = leader
        self._trainer_env = trainer_env

        self._client = DataServerClient()
        self._input_queue = input_queue
        self._cache = {}

        self._meta_queue = threading.Queue()

        self._data_server = DataServer(self)
        self._data_server.start()

        self._connect()

        self._stop = threading.Event()
        self._t_reporter = threading.Thread(target=self._report)
        self._t_accesser = threading.Thread(target=self._access)

    @handle_errors_until_timeout
    def _connect(self, timeout=60):
        self._client.connect(self._leader.endpoint)

    def start(self):
        self._t_reporter.start()
        self._t_accesser.start()

    def _report_and_access(self, report_size=10):
        a = []
        while not self._stop.set():
            while len(a) < report_size:
                b = self._input_queue.pop()
                if b is None:
                    logger.info("data read to end!")
                    break
                a.append(b["id"])

            ret = self._client.get_batch_data_idx(
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
                endpoint=self._leader.endpoint,
                batch_data_ids=a)

            for meta in ret:
                batch_data = self._get_batch_data(meta)
                self._meta_queue.put(meta)

            a = []

        while not self._stop.set():
            self._client.get_batch_data_idx(
                reader_name=self._name,
                pod_id=self.pod_id,
                endpoint=self._data_server.endpoint)

            for meta in ret:
                batch_data = self._get_batch_data(meta)

        # data end
        self._meta_queue.put(None)

    def generate(self):
        while not self._stop.set() and not self._meta_queue.empty():
            meta = self._meta_queue.pop()

            if meta.pod_id != self._trainer_env.pod_id:
                b = self._client.get_batch_data(meta)
            else:
                b = self._cache[meta.batch_data_id]

            self._out_queue.put(b)

    def _access(self):
        try:
            self._report_and_access()
        except EdlDataEndError as e:
            print("reach to data end.")
        except Exception as e:
            raise e


def access_data(self, leader, reader_name, trainer_env, input_queue,
                out_queue):
    try:
        a = DataAccesser(leader, reader_name, trainer_env, input_queue,
                         out_queue)
        a.start()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


class DistributeReader(object):
    def __init__(self,
                 file_list,
                 file_splitter_cls,
                 batch_size,
                 batch_data_cache_size=100):
        self._file_list = file_list
        assert isinstance(self._file_list, list), "file_list must be a list"

        self._name = generator("_dist_reader_")

        self._cls = file_splitter_cls
        self._batch_size = batch_size
        assert self._batch_size > 0, "batch size must > 0"

        # connections to data servers
        self._trainer_env = TrainerEnv()

        self._db = get_global_etcd(trainer_env.endpoints, trainer_env.job_id)
        self._wait_record_to_dist_reader_table()
        self._wait_dist_reader_leader()

        self._generater_out_queue = multiprocessing.Queue()
        self._accesser_out_queue = multiprocessing.Queue()

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
                  self._generater_out_queue, self._accesser_out_queue))
        while True:
            b = self._accesser_out_queue.pop()
            if b is None:
                logger.debug("{} reach data end".format(self._name))
                break
            yield b["meta"], b["data"]
