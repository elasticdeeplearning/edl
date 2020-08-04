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
from paddle_edl.discovery.etcd_client import EtcdClient
import time
import threading
from etcd3.events import PutEvent, DeleteEvent

from paddle_edl.utils.utils import bytes_to_string


class TestRegister(unittest.TestCase):
    def setUp(self):
        self.etcd = EtcdClient()
        self.etcd.init()

    def _register_pod(self):
        pass

    def _verify_pod(self):
        pass

    def _register_master(self):
        pass

    def _verify_master(self):
        pass

    def test_register(self):
        self._register_pod()
        self._verify_pod()

        self._register_master()
        self._verify_master()
