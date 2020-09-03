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
import logging
import google.protobuf.text_format as text_format
import socket
import os
from contextlib import closing

from . import pod_server_pb2
from . import data_server_pb2

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


def get_file_list(file_list):
    """
    return [(file_path, line_no)...]
    """
    line_no = -1
    ret = []
    with open(file_list, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue

            line_no += 1
            ret.append((line, line_no))
    return ret


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


def get_extern_ip():
    return socket.gethostbyname(socket.gethostname())


def get_gpus():
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None or cuda_visible_devices == "":
        selected_gpus = [x.strip() for x in selected_gpus.split(',')]
    else:
        selected_gpus = cuda_visible_devices.split(',')

    logger.info("get selected_gpus:{}".format(selected_gpus))
    return selected_gpus


def bytes_to_string(o, codec='utf-8'):
    if o is None:
        return None

    if not isinstance(o, str):
        return o.decode(codec)

    return o


def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None


def find_free_ports(num):
    if num <= 0:
        return None

    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 100:
            print(
                "can't find avilable port and use the specified static port now!"
            )
            return None

    return None


def handle_timeout_errors(f):
    def handler(*args, **kwargs):
        begin = time.time()
        timeout = kwargs['timeout']
        while True:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if time.time() - begin >= timeout:
                    logger.warning("{} execute timeout:{}".format(f.__name__))
                    raise e

                time.sleep(3)
                continue

    return functools.wraps(f)(handler)
