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
import logging
import google.protobuf.text_format as text_format

logger = logging.getLogger("root")
logger.propagate = False


def get_logger(log_level, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


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
            #print(dataset_to_string(meta))
    return ret


def dataset_to_string(o):
    ret = "data_server:{}, idx_in_list:{}, file_path:{} file_status:{}".format(
        o.data_server, o.idx_in_list, o.file_path, o.file_status)

    ret += " record:["
    for r in o.record:
        ret += "(record_no:{} record_status:{})".format(r.record_no,
                                                        r.record_status)
    ret += "]"

    return ret


def datameta_to_string(o):
    ret = "idx_in_list:{} file_path:{}".format(o.idx_in_list, o.file_path)

    ret += " record_no:["
    for r in o.record_no:
        ret += "(record_no:{})".format(r)
    ret += "]"

    return ret
