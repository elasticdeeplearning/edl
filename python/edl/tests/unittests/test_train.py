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
import edl
from edl.utils.log_utils import logger

def adjust():
    learing_rate = learning_rate * edl.size()

class TestDataReader(etcd_test_base.EtcdTestBase):
    def _read_data(self):
        self._file_list = ["./data_server/a.txt", "./data_server/b.txt"]
        self._data = {}
        for idx, p in enumerate(self._file_list):
            s = TxtFileSplitter(p)
            m = FileMeta()
            for r in s:
                if idx not in d:
                    self._data[idx] = []
                record = ((p), (r[0], r[1:]))
                self._data[idx].append(
                    record)  #[(path),(rec_no, splitted_fiels)]...

    def _train(self, state):
        learning_rate = 1.0
        start_program = None
        main_program = None
        exe = None

        reader = edl.DistributeReader(
            state=state,
            file_list=self._file_list,
            file_splitter_cls=TxtFileSplitter,
            batch_size=1)

        for epoch in range(state.epoch, 5):
            for meta, batch in reader:
                print("epoch_no:",epoch)
                edl.notify_end_one_batch(meta, state)
            edl.notify_end_one_epoch(state)

    def test_data_reader(self):
        state = edl.PaddleState(
            exe, start_program, main_program, optimizer, max_epoch_num=5)
        state.register_adjust_function([adjust])
        self._train(state)


if __name__ == '__main__':

    unittest.main()
