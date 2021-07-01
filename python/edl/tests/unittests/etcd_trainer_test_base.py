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

import copy
import edl.utils.constants as constants
import edl.utils.log_utils as log_utils
import os
import unittest
from edl.discovery.etcd_client import EtcdClient

g_etcd_endpoints = "127.0.0.1:2379"


class EtcdTestBase(unittest.TestCase):
    def setUp(self, job_id):
        log_utils.get_logger(log_level=10)
        self._etcd = EtcdClient([g_etcd_endpoints], root=job_id)
        self._etcd.init()

        self._old_environ = copy.copy(dict(os.environ))
        constants.clean_etcd(self._etcd)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)
        constants.clean_etcd(self._etcd)
