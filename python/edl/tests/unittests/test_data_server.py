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

import numpy as np
from edl.collective.data_reader import DistributedDataReader, FileMeta
from edl.collective.dataset import TxtFileSplitter
import unittest
from edl.utils import data_server
from edl.utils import data_server_client
from edl.utils import edl_env


class TestDataServer(unittest.TestCase):
    def setUp(self):
        self._trainer_env = edl_env.TrainerEnv()
        self._file_list = ["./data_server/a.txt", "./data_server/b.txt"]
        self._reader_name = "test_data_server_reader"
        self._pod_id = "test_data_server_pod_id"
        self._data_server = data_server.DataServer(
            trainer_env=self._trainer_env,
            reader_name=self._reader_name,
            file_list=self._file_list,
            local_data=None)

    def test_get_file_list(self):
        c = data_server_client.Client()
        res = c.get_file_list(self._data_server.endpoint, self._reader_name,
                              self._pod_id, self._file_list)
        self.assertTrue(len(res.file_list, len(self._file_list)))

        for l in self._file_list:
            self.assertTrue(l in res.file_list)

    def test_report_batch_data_ids(self):
        pass

    def test_get_batch_data_ids(self):
        pass

    #TODO(gongwb): add test_get_batch_data
    """
    def test_get_batch_data(self):
        pass
    """
