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

import sys
from edl.utils import data_server_client
from edl.utils import data_server_pb2
from edl.utils import edl_process


class Generator(edl_process.ProcessWrapper):
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
        client = data_server_client.DataServerClient()
        return client.get_file_list(
            leader_endpoint=self._leader_endpoint,
            reader_name=self._reader_name,
            pod_id=self._pod_id,
            file_list=self._file_list)

    def _generate_batch_data(self):
        self._batch_data_id += 1
        b = data_server_pb2.BatchData()
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
                rec = data_server_pb2.Record()
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