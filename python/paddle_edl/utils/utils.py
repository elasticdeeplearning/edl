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

import master_pb2
import data_server_pb2


def file_list_to_dataset(file_list):
    line_no = -1
    ret = []
    with open(file_list, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue

            line_no += 1
            meta = master_pb2.FileDataSet()
            meta.idx_in_list = line_no
            meta.file_path = line
            meta.file_status = master_pb2.ProcStatus.INITIAL
            ret.append(meta)
    return ret
