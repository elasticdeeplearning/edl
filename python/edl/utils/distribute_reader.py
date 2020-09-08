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


class Connection(self):
    def __init__(self, endpoint, channel, stub):
        self._endpoint = endpoint
        self.channel = channel
        self.stub = stub


class DataGenerator(ProcessWrapper):
    def __init__(self,
                 endpoint,
                 total_file_list,
                 splitter_cls,
                 data_queue,
                 error_queue=None):
        super(DataGenerator, self).__init__()

        self._batch_data_id = 0

        self._file_list = total_file_list
        self._splitter_cls = splitter_cls
        self._data_queue = data_queue
        self._error_queue = error_queue

        self._client = DataServerClient()

    def start(self):
        super(DataGenerator, self).start()
        self._client.connect(endpoint)

    @handle_errors_until_timeout
    def _get_file_list(self, timeout=120):
        return self._client.get_file_list(self._reader_leader._endpoint,
                                          self._name, self._trainer_env.pod_id)

    def _generate_batch_data(self):
        self._batch_data_id += 1
        return {
            "id:": self._batch_data_id,
            #"producer_pod_id":None,
            #"consumer_pod_id":None,
            "meta": None,  # {idx=>record_nos, }
            "data": None,  # [(field_data...) ...]
        }

    def _read_batch_data(self):
        """
        #  load checkpoint from etcd 
        self._checkpoint = Checkpoint.load_from_etcd(
                etcd_endpoints=self._trainer_env.etcd_endpoints,
                job_id=self._trainer_env.job_id,
                reader_name=self._name)
        """

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


class DataReporter(object):
    def __init__(self, leader, reader_name, input_queue, out_queue,
                 trainer_env):
        super(DataGenerator, self).__init__()

        self._reader_name = reader_name
        self._trainer_env = trainer_env
        self._leader = leader
        self._client = DataServerClient()
        self._input_queue = input_queue
        self._cache = {}

        self._data_server = DataServer(self)
        self._data_server.start()

        self._t_reporter = threading.Thread(target=self._report)
        self._t_reporter.start()

        self._t_getter = threading.Thread(target=self._report)
        self._t_getter.start()

    def start(self):
        try:
            super(DataGenerator, self).start()
            self._client.connect(self._leader.endpoint)
        except Exception as e:
            print(e, file=stderr)
            sys.exit(1)

    def _report(self):
        while not self._stop.set():
            b = self._input_queue.pop()
            if b is None:
                logger.info("data read to end!")
                break

            ret = self._client.get_batch_data_idx(
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
                endpoint=self._leader.endpoint,
                batch_data_ids=[b["id"]])

    def _getter(self):
        self._client.get_batch_data_idx(
            reader_name=self._name,
            pod_id=self._trainer_env.pod_id,
            endpoint=self._data_server.endpoint)

        for meta in ret:
            batch_data = self._get_batch_data(meta)

    def get_data(self, meta):
        conn = self._connect_data_server(endpoint)
        response = conn.stub.GetData(request)

        if len(response.errors.errors) > 1:
            return

        data = []
        for f in response.files.files:
            for rec in f.records:
                data.append(rec.data)

        return data


def data_report(self):
    try:
        reporter = DataReporter()
        reporter.start()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


class DistributeReader(object):
    def __init__(self, file_list, file_splitter_cls, batch_size):
        self._file_list = file_list
        assert isinstance(self._file_list, list)

        self._name = generator("_dist_reader_")

        self._cls = file_splitter_cls
        self._batch_size = batch_size
        assert self._batch_size > 0

        # connections to data servers
        #self._conns = {}
        self._trainer_env = TrainerEnv()

        self._db = get_global_etcd(trainer_env.endpoints, trainer_env.job_id)
        self._wait_dist_reader_leader()

        self._generater_out_queue = multiprocessing.Queue()
        self._reporter_out_queue = multiprocessing.Queue()

        self._generater = None
        self._reporter = None

    @handle_errors_until_timeout
    def _wait_dist_reader_leader(self, timeout=120):
        self._reader_leader = self._db.get_dist_reader_leader()

    def stop(self):
        if self._generater:
            self._generater.stop()
            self._generater = None

        if self._reporter:
            self._reporter.terminate()
            self._reporter = None

    def __exit__(self):
        self.stop()

    def __iter__(self):
        self._generater = DataGenerator()
        self._generator.start()

        self._reporter = multiprocessing.Process(data_report)
        self._reporter.start()

        while not self._reporter_out_queue.empty():
            yield self._reporter_out_queue.pop()
