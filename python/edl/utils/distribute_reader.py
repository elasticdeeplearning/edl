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

import multiprocessing
import threading
import sys

from . import data_server_pb2 as pb
from .data_server_client import DataServerClient
from .etcd_db import get_global_etcd
from .log_utils import logger
from .unique_name import generator
from .edl_env import TrainerEnv
from .error_utils import handle_errors_until_timeout
from . import edl_process
from . import data_server


class DataGenerator(edl_process.ProcessWrapper):
    """
    1. get file_list from data_server_leader
    2. parse files of file_list and put BatchData to out_quque
       if reach data end, put None to out_queue.
    3. program will exit if meets any error
    """

    def __init__(self, reader_leader_endpoint, reader_name, pod_id,
                 all_files_list, splitter_cls, out_queue):
        super(DataGenerator, self).__init__()

        self._batch_data_id = 0

        self._leader_endpoint = reader_leader_endpoint
        self._pod_id = pod_id
        self._reader_name = reader_name

        self._file_list = all_files_list
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
        b = self._generate_batch_data()
        for m in self._get_file_list():
            if self._stop.set():
                break

            assert self._file_list[m.idx] == m.path
            for record in self._splitter_cls(m.path):
                fields = record

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
    def __init__(self, reader_leader_endpoint, reader_name, trainer_env,
                 input_queue, out_queue, queue_size):
        self._reader_leader_endpoint = reader_leader_endpoint
        self._data_server_endpoint = data_server_endpoint
        self._reader_name = reader_name
        self._trainer_env = trainer_env

        # BatchData
        self._input_queue = input_queue
        self._out_queue = out_queue
        # batch_data_id => BatchData
        self._cache = {}

        # pb.BatchDataRequest queue
        self._req_queue = threading.Queue(queue_size)

        self._data_server = data_server.DataServer(self)
        self._data_server.start()

        self._stop = threading.Event()
        self._t_reporter = threading.Thread(target=self.report)
        self._t_generater = threading.Thread(target=self.generate)
        self._t_accesser = threading.Thread(target=self.access)

        self._client = DataServerClient()

    def start(self):
        self._client.connect(self._reader_leader_endpoint)
        self._t_reporter.start()
        self._t_generater.start()
        self._t_accesser.start()

    def _report(self, report_size=10):
        """
        1. Report BatchData index to Leader
        2. Get the BatchData index need to be processed
            if there is no data, set None to req_queue
        """
        batch_data_ids = []
        while not self._stop.set():
            while len(a) < report_size:
                b = self._input_queue.pop()
                if b is None:
                    logger.info("data read to end!")
                    break
                batch_data_ids.append(b.batch_data_id)
                with self._lock:
                    self._cache[b.batch_data_id] = b

            self._client.report_batch_data_meta(
                reader_leader_endpoint=self._reader_leader_endpoint,
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
                dataserver_endpoint=self._data_server.endpoint,
                batch_data_ids=batch_data_ids)

            batch_data_ids = []

        while not self._stop.set() and len(batch_data_ids) > 0:
            self._client.report_batch_data_meta(
                reader_leader_endpoint=self._reader_leader_endpoint,
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
                dataserver_endpoint=self._data_server.endpoint,
                batch_data_ids=batch_data_ids)

        self._client.reach_data_end(
            reader_leader_endpoint=self._reader_leader_endpoint,
            reader_name=self._name,
            pod_id=self._trainer_env.pod_id)

    def _access(self):
        while not self._stop.set():
            res = self._client.get_balanced_batch_data(
                reader_leader_endpoint=self._reader_leader_endpoint,
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id)

            self._req_queue.put(res)

            # data end
            if res is None:
                break

    def _get_batch_data(self, req):
        """
        Read BatchData from local or remote by BatchDataRequest
        """
        if self._trainer_env.pod_id != req.producer_pod_id:
            return (req, self._client.get_batch_data(req))

        return (req, self.get_local_batch_data(req))

    def get_local_batch_data(self, req):
        ret = []
        for batch_data_id in req.data.batch_data_ids:
            with self._lock:
                ret.append(self._cache.pop(batch_data_id))

        return ret

    def _generate(self):
        while not self._stop.set():
            req = self._req_queue.pop()
            if req is None:
                break

            ret = self._get_batch_data(req)
            for b in ret:
                self._out_queue.put(b)

        self._out_queue.put(None)

    def report(self):
        try:
            self._report()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def access(self):
        try:
            self._access()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def generate(self):
        try:
            self._generate()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)


def access_batch_data(reader_leader, reader_name, trainer_env, input_queue,
                      out_queue, cache_size):
    """
    Run DataAccesser in a seperated process
    """
    try:
        a = DataAccesser(reader_leader, reader_name, trainer_env, input_queue,
                         out_queue, cache_size)
        a.start()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


class Reader(object):
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

        self._db = get_global_etcd(self._trainer_env.endpoints,
                                   self._trainer_env.job_id)
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
            access_batch_data,
            args=(self._reader_leader, self._name, self._trainer_env,
                  self._generater_out_queue, self._accesser_out_queue,
                  self._cache_size))
        while True:
            b = self._accesser_out_queue.pop()
            if b is None:
                logger.debug("{} reach data end".format(self._name))
                break
            yield {"meta": b[0], "data": b[1]}
