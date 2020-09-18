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

class UserDefined(object):
    def __init__(self):
        self.learning_rate = 1.11

    def from_json(self, json_str):
        d = json.loads(json_str)
        self.learning_rate = d["learning_rate"]

    def to_json(self):
        d ={
            "learning_rate": self.learning_rate
        }
        return json.dumps(d)


class TestState(EtcdTestBase):
    def setUp(self):
        super(TestState, self).setUp("test_state")

    def test_state(self):
        user_defined =UserDefined()

        state = edl_state.State(total_batch_size=1000, user_defined=user_defined)
        state._model_path = "model_path"

        # data checkpoint
        data_checkpoint = edl_state.DataCheckpoint()
        state._data_checkpoint = data_checkpoint

        train_status = edl_state.TrainStatus()
        train_status.epoch_no = 1
        train_status.global_step_no = 2
        epoch_attr = train_status.get_current_epoch_attr()
        epoch_attr.epoch_no = train_status.epoch_no

        state._train_status = train_status

        # save
        edl_state.save_to_etcd(self._etcd, "0", state)

        # load
        state2 = edl_state.load_from_etcd(self._etcd, state.name)
        self.assertEqual(state, state2)