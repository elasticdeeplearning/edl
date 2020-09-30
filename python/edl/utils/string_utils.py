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


def dataset_to_string(o):
    """
    FileMeta to string
    """
    ret = "idx_in_list:{}, file_path:{}".format(o.idx_in_list, o.file_path)

    ret += " record:["
    for rs in o.records:
        for rec_no in range(rs.begin, rs.end + 1):
            ret += "(record_no:{})".format(rec_no)
    ret += "]"

    return ret


def data_request_to_string(o):
    """
    DataMeta to string
    """
    ret = "idx_in_list:{} file_path:{}".format(o.idx_in_list, o.file_path)
    for rs in o.chunks:
        ret += " chunk:["
        ret += chunk_to_string(rs)
        ret += "]"

    return ret


def chunk_to_string(rs):
    ret = "status:{} ".format(rs.status)
    for rec_no in range(rs.meta.begin, rs.meta.end + 1):
        ret += "(record_no:{}) ".format(rec_no)

    return ret


def bytes_to_string(o, codec="utf-8"):
    if o is None:
        return None

    if not isinstance(o, str):
        return o.decode(codec)

    return o
