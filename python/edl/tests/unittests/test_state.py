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

import json
import unittest
from edl.collective import serializable
from edl.tests.unittests import etcd_test_base
from edl.utils import constants
from edl.collective import state as edl_state


class UserDefined(serializable.SerializableBase):
    def __init__(self):
        self.learning_rate = 1.11

    def from_json(self, json_str):
        d = json.loads(json_str)
        self.learning_rate = d["learning_rate"]

    def to_json(self):
        d = {"learning_rate": self.learning_rate}
        return json.dumps(d)


class TestState(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestState, self).setUp("test_state")

    def _generate_train_status(self):
        train_status = edl_state.TrainStatus()
        train_status.epoch_no = 1
        train_status.global_step_no = 2

        epoch_attr = edl_state.EpochAttr()
        epoch_attr.epoch_no = 1
        epoch_attr.world_size = 1
        epoch_attr.step_num = 10
        epoch_attr.avg_step_time = 100
        epoch_attr.step_no_of_epoch = 5

        train_status.update_epoch_attr(epoch_attr.epoch_no, epoch_attr)

        return train_status

    def _generate_data_checkpoint(self):
        dp = edl_state.DataCheckpoint()
        dp.reader_name = "reader"
        dp.file_list = ["0", "1"]
        dp.processed_data = {"0": [[0, 1], [2, 3]], "1": [[4, 5], [6, 7]]}

        return dp

    def test_state(self):
        user_defined = UserDefined()

        state = edl_state.State(total_batch_size=1000, user_defined=user_defined)
        state._model_path = "model_path"
        state._data_checkpoint = self._generate_data_checkpoint()
        state._train_status = self._generate_train_status()

        print("state", state)

        # save
        pod_id = "0"
        self._etcd.set_server_permanent(
            constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER, pod_id
        )
        edl_state.save_to_etcd(self._etcd, pod_id, state, timeout=10)

        # load
        state2 = edl_state.load_from_etcd(
            self._etcd, state.name, user_defined=user_defined, timeout=10
        )
        print("state2", state2)

        # compare
        self.assertEqual(state, state2)

        # only leader can write state
        try:
            pod_id = "1"
            self._etcd.set_server_permanent(
                constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER, pod_id
            )
            edl_state.save_to_etcd(self._etcd, pod_id, state, timeout=10)
            self.assertFalse(True)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
