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
import time
import os
import atexit

from edl.utils.global_vars import *
from edl.utils.etcd_test_base import EtcdTestBase
from edl.utils import state as edl_state


class TestState(EtcdTestBase):
    def setUp(self):
        super(TestState, self).setUp("test_state")

    def test_state(self):
        edl_state.save_to_etcd()
        new_state = edl_state.load_from_etcd()
        self.asserEqual(old_state, new_state)