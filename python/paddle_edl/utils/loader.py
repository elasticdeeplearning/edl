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

import paddle.fluid as fluid
from fluid.reader import DataLoaderBase, GeneratorLoader


class Loader(DataLoaderBase):
    def __init__(dataset,
                 feed_list=None,
                 places=None,
                 return_list=False,
                 batch_sampler=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 collate_fn=None,
                 num_workers=0,
                 use_buffer_reader=True,
                 use_shared_memory=True,
                 timeout=0,
                 worker_init_fn=None):
        self._name = unique_name.generate()

        self._start_data_server()

    def _start_data_server(self):
        """
        start and register the data server
        """
        pass

    def _reader(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass
