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
import paddle_edl.distill.distill_reader as distill_reader

if __name__ == '__main__':
    # temp local test
    distill_reader._NOP_PREDICT_TEST = True

    # test mnist distill reader
    def _reader():
        img = np.array(
            [(i + 1) / 28.0 for i in range(28)] * 28,
            dtype=np.float32).reshape((1, 28, 28))
        label = np.array([100], dtype=np.int64)
        for i in range(24):
            yield 8 * [(img, label)]
        yield 2 * [(img, label)]

    dr = distill_reader.DistillReader(
        'distill_reader_test.conf', 32, 4, capacity=4, occupied_capacity=2)
    dr.set_sample_list_generator(_reader)
    train_reader = dr.distill_reader()

    for epoch in range(300):
        for step, batch in enumerate(train_reader()):
            # print('----step={}, predict_shape={}, predict[0]={} ----'.format(step, batch[-1].shape, batch[-1][0]))
            pass
        if epoch % 10 == 0:
            print('^^^^^^^^^^^^^ epoch={} predict[0][0]={}^^^^^^^^^^^^^^'.
                  format(epoch, batch[-1][0][0]))

    fake_dr = distill_reader.FakeDistillReader('distill_reader_test.conf')
    fake_test_reader = fake_dr.fake_from_sample_list_generator(_reader)
    for epoch in range(20):
        for step, sample_list in enumerate(fake_test_reader()):
            # print('---step={}, predict_shape={}, predict[0]={}---'.format(step, sample_list[0][-1].shape, sample_list[0][-1][0]))
            pass
        if epoch % 10 == 0:
            print('^^^^^^^^^^^^^ fake_epoch={} predict[0][0]={}^^^^^^^^^^^^^^'.
                  format(epoch, sample_list[0][-1][0]))
