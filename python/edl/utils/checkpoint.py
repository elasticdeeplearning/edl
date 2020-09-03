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
import threading
import time
import json
import uuid
import copy
import traceback
import six

from .utils import logger
from .pod import Pod
from ..discovery.etcd_client import EtcdClient

import etcd3
from .global_vars import *
from .cluster import Cluster
from .exceptions import EdlPutError
from .etcd_db import get_global_etcd


class Checkpoint(object):
    def __init__(self):
        self._reader_name = None
        self._model_path = None
        self._data_checkpoint = None

    def to_json(self):
        d = {
            "reader_name": self._reader_name,
            "model_path": self._model_path,
            "data_checkpoint": self._data_checkpoint,
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)
        self._reader_name = d["reader_name"]
        self._model_path = d["model_path"]
        self._data_checkpoint = d["data_checkpoint"]
        return d

    @staticmethod
    @handle_errors_until_timeout
    def load_from_etcd(etcd_endpoints, job_id, reader_name):
        etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=EDL_CONN_TIMEOUT)
        etcd.init()

        value = etcd.get_value(ETCD_DIST_READER, reader_name)

        if value is None:
            raise EdlTableError("key:value = {}:{}".format(
                etcd.get_full_path(ETCD_DIST_READER, reader_name), value))

        c = Checkpoint()
        c.from_json(value)
        return c

    @staticmethod
    @handle_errors_until_timeout
    def save_to_etcd(etcd_endpoints, job_id, pod_id, reader_name, mode_path,
                     data_checkpoint):
        etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=EDL_CONN_TIMEOUT)
        etcd.init()

        c = Checkpoint()
        c._reader_name = reader_name
        c._data_checkpoint = data_checkpoint
        c._model_path = model_path

        leader_key = etcd.get_full_path(ETCD_POD_RANK, ETCD_POD_LEADER)
        dist_reader_key = etcd.get_full_path(ETCD_DIST_READER, reader_name)

        etcd = etcd._etcd
        status, _ = etcd.transaction(
            compare=[etcd.transactions.value(leader_key) == pod_id, ],
            success=[etcd.transactions.put(dist_reader_key, c.to_json()), ],
            failure=[])

        message = "pod_id:{} save_data_checkpoint status:{}".format(pod_id,
                                                                    status)

        if not status:
            raise EdlPutError(message)

        return status
