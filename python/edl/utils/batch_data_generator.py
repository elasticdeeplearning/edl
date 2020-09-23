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
import os
from edl.utils import data_server_client
from edl.utils import data_server_pb2
from edl.utils import edl_process
from edl.utils import log_utils

logger = None

class Args(object):
    def __init__(self):
        self.state = None
        self.reader_leader_endpoint= None
        self.reader_name= None
        self.pod_id= None
        self.all_files_list= None
        self.splitter_cls= None
        self.out_queue= None
        self.error_queue= None

class Generator(edl_process.ProcessWrapper):
    """
    1. get file_list from data_server_leader
    2. parse files of file_list and put BatchData to out_quque
       if reach data end, put None to out_queue.
    3. program will exit if meets any error
    """

    def __init__(self, state, reader_leader_endpoint, reader_name, pod_id,
                 all_files_list, splitter_cls, out_queue, error_queue):
        self._state = state
        self._batch_data_id = 0

        self._leader_endpoint = reader_leader_endpoint
        self._reader_name = reader_name
        self._pod_id = pod_id

        self._file_list = all_files_list
        self._splitter_cls = splitter_cls
        self._data_queue = out_queue
        self._error_queue = error_queue
        self._batch_data_ids = []

    def _get_file_list(self, timeout=60):
        client = data_server_client.Client()
        return client.get_file_list(
            leader_endpoint=self._leader_endpoint,
            reader_name=self._reader_name,
            pod_id=self._pod_id,
            file_list=self._file_list)

    def _generate_batch_data(self):
        self._batch_data_id += 1
        b = data_server_pb2.BatchData()
        b.batch_data_id = self._batch_data_id
        b.records = None

        return b

    def _report(self, batch_data, report_size=10):
        if batch_data is None:
            if len(self._batch_data_ids) > 0:
                self._client.report_batch_data_meta(
                    reader_leader_endpoint=self._reader_leader_endpoint,
                    reader_name=self._name,
                    pod_id=self._trainer_env.pod_id,
                    dataserver_endpoint=self._data_server.endpoint,
                    batch_data_ids=batch_data_ids)
                return

        if len(self._batch_data_ids) <= report_size - 1:
            self._batch_data_ids.append(batch_data.batch_data_id)
            return

        self._client.report_batch_data_meta(
            reader_leader_endpoint=self._reader_leader_endpoint,
            reader_name=self._name,
            pod_id=self._trainer_env.pod_id,
            dataserver_endpoint=self._data_server.endpoint,
            batch_data_ids=self._batch_data_ids)
        self._batch_data_ids = []

    def _read_batch_data(self):
        batch_data = self._generate_batch_data()
        for ele in self._get_file_list():
            if self._stop.set():
                break

            assert self._file_list[ele.idx] == ele.path
            logger.info("begin process file {}:{}".format(ele.idx, ele.path))

            for record in self._splitter_cls(ele.path):
                fields = record

                rec = data_server_pb2.Record()
                rec.record_no = fields[0]
                assert isinstance(rec.record_no,int), \
                    "first element of splitter_cls must be the record index of this file"

                #FIXME(gongwb) filter it
                for field in fields[1:]:
                    rec.field_data.append(field)
                batch_data.records.append(rec)

                if len(batch_data.records) >= self._batch_size:
                    yield batch_data
                    batch_data = self._generate_batch_data()

        if len(batch_data.records) > 0:
            yield batch_data

    def read_batch_data(self):
        for batch_data in self._read_batch_data():
            self._report(batch_data)
            self._data_queue.put(batch_data)

        self._report(None)
        self._data_queue.put(None)
        self._client.reach_data_end(
            reader_leader_endpoint=self._reader_leader_endpoint,
            reader_name=self._name,
            pod_id=self._trainer_env.pod_id)

def generate(args)
    log_file_name = "edl_data_generator_{}.log".format(os.getpid())
    global logger
    logger = log_utils.get_logger(log_level=20, log_file_name=log_file_name)

    cls = Generator()
    try:
        cls.read_batch_data()
    except:
        import traceback
        args.error_queue.put(traceback.format_exc())