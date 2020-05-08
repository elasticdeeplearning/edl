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

import unittest
from paddle_edl.utils.data_server import DataServer
from paddle_edl.utils.dataset import TxtDataSet
import paddle_edl.utils.master_pb2_grpc as master_pb2_grpc
import paddle_edl.utils.master_pb2 as master_pb2
from paddle_edl.utils.utils import file_list_to_dataset, get_logger
import time
import threading
import grpc
import signal
import paddle_edl.utils.common_pb2 as common_pb2
import os

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""


class TestMasterClient(unittest.TestCase):
    def setUp(self):
        self._client = Client()
        pass

    def test_add_dataset(self):
        pass
