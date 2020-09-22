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

from edl.utils.etcd_test_base import EtcdTestBase
from edl.utils.global_vars import *


class TestDistReader(EtcdTestBase):
    def setUp(self):
        super(TestGenerate, self).__init__("test_dist_reader")

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

    # 1. start a server, load checkpoint from etcd
    # 2. get records from it 
    def test_base(self):
        reader1 = DistributedDataReader(
            file_list=["./data_server/a.txt"],
            file_splitter_cls=TxtFileSplitter,
            splitted_data_field=["line"],
            batch_size=1)

        pass

    # 1. start two servers and read their file list and the leader loads checkpoint from etcd
    # 2. leader can balance them 
    def test_balance(self):
        pass
