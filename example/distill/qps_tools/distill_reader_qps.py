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
import datetime
import time

from paddle_edl.distill.distill_reader import DistillReader
from parse_config import get_ins_predicts


def sample_reader(shapes, dtypes, sample_num=1 << 12):
    def __reader_impl__():
        for _ in range(sample_num):
            sample = tuple()
            for shape, dtype in zip(shapes, dtypes):
                sample += (np.random.random(shape).astype(dtype), )
            yield sample

    return __reader_impl__


def qps(reader):
    pre_t = time.time()
    for step, _ in enumerate(reader()):
        if (step + 1) % 1000 == 0:
            now = datetime.datetime.now()
            t = time.time()
            print('{}, step={}, qps={} step/s'.format(now, step + 1, 1000.0 / (
                t - pre_t)))
            pre_t = t


def main(args):
    ins, ins_shape, ins_dtype, predicts = get_ins_predicts()
    print('{}, {}, {}, {}'.format(ins, ins_shape, ins_dtype, predicts))

    reader = sample_reader(ins_shape, ins_dtype, 1 << 12)

    dr = DistillReader(ins=ins, predicts=predicts)
    dr.set_teacher_batch_size(args.teacher_bs)
    #dr.set_fixed_teacher(['10.255.100.13:9494'])
    distill_reader = dr.set_sample_generator(reader)

    qps(distill_reader)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='qps test')
    parser.add_argument(
        '--teacher_bs',
        type=int,
        default=1,
        help='teacher batch_size [default: %(default)s]')
    args = parser.parse_args()
    main(args)
