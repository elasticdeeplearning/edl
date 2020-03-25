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

import os
import cPickle
import paddle
import glob


def prepare_dataset(output_path, name_prefix, reader_func, sample_count=128):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    suffix = "%s/%s-%%05d.pickle" % (output_path, name_prefix)
    lines = []
    indx_f = 0
    for i, d in enumerate(reader_func()):
        lines.append(d)
        if i >= sample_count and i % sample_count == 0:
            with open(suffix % indx_f, "w") as f:
                cPickle.dump(lines, f)
                lines = []
                indx_f += 1
    if lines:
        with open(suffix % indx_f, "w") as f:
            cPickle.dump(lines, f)


def cluster_reader(files_path, trainers, trainer_id):
    def reader():
        flist = glob.glob(files_path)
        flist.sort()
        my_file_list = []
        for idx, fn in enumerate(flist):
            if idx % trainers == trainer_id:
                print("append file for current trainer: %s" % fn)
                my_file_list.append(fn)

        for fn in my_file_list:
            print("processing file: ", fn)
            with open(fn, "r") as f:
                lines = cPickle.load(f)
                for line in lines:
                    yield line

    return reader
