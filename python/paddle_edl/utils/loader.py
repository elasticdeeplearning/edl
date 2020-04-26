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
    def __init__(dataset, feed_list, batch_size, capcity=None, master=None):
        self._batch_size = batch_size
        self._capcity = capcity
        self._generator = GeneratorLoader(feed_list=feed_list, capcity=capcity)
        self._data_server = DataServer(self._master)
        self._meta = MetaReader(master=master)
        self._data_server.start()

    def _get_data(self, meta):
        pass

    def _reader(self):
        metas = self._meta.get_meta(self._batch_size, step_num=self.capcity)
        for meta in metas:
            try:
                data = self._get_data(meta)
                yield data
            except NetworkException as e:
                self._meta.report(meta, False)
                continue

    def __iter__(self):
        self._generator.set_sample_generator(self._reader)
        return self._generator.__iter__()

    def __next__(self):
        return self._generator.__next__()

    def next(self):
        return self._generator.next()
