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
from edl.data_reader import DataReader
from edl.dataset import TxtFileSplitter, FileMeta
#from paddle.fluid.incubate.fleet.collective import fleet
import unittest

learning_rate = 1.0
start_program = None
main_program = None
exe = None


def on_state_reset():
    learing_rate = learning_rate * edl.size()


class TestDataReader(unittest.TestCase):
    def setUp(self):
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

    @edl.run
    def _train(self, state):
        print("learning_rate:", learning_rate)
        reader = DataReader(
            file_list=file_list,
            file_splitter_cls=TxtFileSplitter,
            splitted_data_field=["line"],
            batch_size=1,
            trainer_rank=0)

        for epoch in range(state.epoch, 5):
            for meta, batch in reader:
                edl.notify_end_one_batch(meta, state)
            edl.notify_end_one_epoch(meata, state)

    def test_data_reader(self):
        state = edl.PaddleState(
            exe, start_program, main_program, optimizer, batch=0, epoch=0)
        state.register_reset_callbacks([on_state_reset])
        train(state)


if __name__ == '__main__':
    unittest.main()
