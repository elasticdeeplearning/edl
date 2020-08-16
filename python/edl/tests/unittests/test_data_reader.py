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
                self._data[idx].append(
                    (p), (r[0], r[1:]))  #[(path),(rec_no, splitted_fiels)]...

    def test_data_reader(self):
        reader1 = DataReader(
            file_list=file_list,
            file_splitter_cls=TxtFileSplitter,
            splitted_data_field=["line"],
            batch_size=1,
            trainer_rank=0)

        reader2 = DataReader(
            file_list=file_list,
            file_splitter_cls=TxtFileSplitter,
            splitted_data_field=["line"],
            batch_size=1,
            trainer_rank=1)

        size1 = 0
        for meta, batch in reader1:
            self.assertTrue(meta._size, 1)
            for k, v in meta._batch:
                c = self._data[k._idx]
                self.assertTrue(c[0][0], k._path)
                size1 += 1

        for meta, batch in reader2:
            self.assertTrue(meta._size, 1)
            for k, v in meta._batch:
                c = self._data[k._idx]
                self.assertTrue(c[0][0], k._path)
                size2 += 1

        self.assertTrue(size1, size2)


if __name__ == '__main__':
    unittest.main()
