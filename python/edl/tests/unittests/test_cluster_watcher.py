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
import time
import unittest
import six

from edl.utils import constants
from edl.utils import resource_pods
from edl.tests.unittests import etcd_test_base
from edl.utils import cluster as edl_cluster


class TestWatcher(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestWatcher, self).setUp("test_watcher")

    def test_watcher(self):

