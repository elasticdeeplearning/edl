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
from edl.collective.data_reader import DistributedDataReader, FileMeta
from edl.collective.dataset import TxtFileSplitter


class TestDataReader(unittest.TestCase):
    def setUp(self):
        self._file_list = ["./data_server/a.txt", "./data_server/b.txt"]
        self._data = {}
        for idx, p in enumerate(self._file_list):
            s = TxtFileSplitter(p)
            m = FileMeta(idx, p)
            for r in s:
                if idx not in self._data:
                    self._data[idx] = []
                d = ((p), (r[0], r[1:]))
                self._data[idx].append(
                    d)  #[(path),(rec_no, splitted_fiels)]...

    def test_data_reader(self):
        reader1 = DistributedDataReader(
            file_list=self._file_list,
            file_splitter_cls=TxtFileSplitter,
            splitted_data_field=["line"],
            batch_size=1)

        reader2 = DistributedDataReader(
            file_list=self._file_list,
            file_splitter_cls=TxtFileSplitter,
            splitted_data_field=["line"],
            batch_size=1)

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
