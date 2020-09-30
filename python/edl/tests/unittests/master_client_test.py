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

import os
import paddle_edl.utils.master_pb2 as master_pb2
import unittest
from edl.utils.master_client import Client
from edl.utils.utils import get_file_list, get_logger

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""


class TestMasterClient(unittest.TestCase):
    def setUp(self):
        self._client = Client("127.0.0.1:8080")

    def test_add_dataset(self):
        dataset = master_pb2.DataSet()
        dataset.name = "train"
        for t in get_file_list("./test_file_list.txt"):
            dataset.file_list.append(t[0])

        res = self._client.add_dataset(dataset)
        assert res is None or res.type == "", "must not any error"

        res = self._client.add_dataset(dataset)
        assert res.type == "DuplicateInitDataSet", "must error"


if __name__ == "__main__":
    logger = get_logger(10)
    unittest.main()
