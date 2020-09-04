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
from enum import IntEnum

from .utils import logger
from .pod import Pod
from ..discovery.etcd_client import EtcdClient

import etcd3
from .global_vars import *
from .cluster import Cluster
from .exceptions import EdlPutError
from .etcd_db import get_global_etcd


class State(object):
    def __init__(self, total_batch_size, user_defined=None):
        # interface
        self._default = {
            "total_batch_size": total_batch_size,
            "epoch_num": None,
            "epoch_no": None,
            "global_step_no": None,
            "step_no_of_epoch": None,
        }
        self._user_defined = user_defined
        self._adjust_func = []

        # internal
        self._name = generator("_edl_state_")
        self._model_path = None
        self._data_checkpoint = DataCheckpoint()
        self._train_status = TrainStatus()

    def register_adjust_function(self, f):
        self._adjust_func.append(f)

    @property
    def epoch_num(self):
        return self._defaults["epoch_num"]

    @property
    def epoch_no(self):
        return self._defaults["epoch_no"]

    @property
    def step_no_of_epoch(self):
        return self._defaults["step_no_of_epoch"]

    @property
    def global_step_no(self):
        return self._defaults["global_step_no"]

    @property
    def total_batch_size(self):
        return self._defaults["total_batch_size"]

    @total_batch_size.setter
    def total_batch_size(self, size):
        self._defaults["total_batch_size"] = size

    def to_json(self):
        d = {
            "default": json.to_json(self._default),
            "user_defined": self._user_defined.to_json()
            if self._user_defined else None,
            "name": self._name,
            "model_path": self._model_path,
            "data_checkpoint": self._data_checkpoint.to_json(),
            "train_status": self._train_status.to_json(),
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)

        self._defaults = d["default"]
        if self._user_defined and d["user_defined"] is not None:
            self._user_defined.from_json(d["user_defined"])

        self._name = d["name"]
        self._model_path = d["model_path"]
        self._data_checkpoint.from_json(d["data_checkpoint"])
        self._train_status.from_json(d["train_status"])
        return d

    @staticmethod
    @handle_errors_until_timeout
    def load_from_etcd(etcd_endpoints, job_id, name):
        etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=EDL_CONN_TIMEOUT)
        etcd.init()

        value = etcd.get_value(ETCD_DIST_READER, name)

        if value is None:
            raise EdlTableError("key:value = {}:{}".format(
                etcd.get_full_path(ETCD_DIST_READER, name), value))

        c = State()
        c.from_json(value)
        return c

    @staticmethod
    @handle_errors_until_timeout
    def save_to_etcd(etcd_endpoints, job_id, pod_id, name, mode_path,
                     data_checkpoint, user_defined):
        etcd = EtcdClient(
            endpoints=etcd_endpoints, root=job_id, timeout=EDL_CONN_TIMEOUT)
        etcd.init()

        c = State()
        c._name = name
        c._data_checkpoint = data_checkpoint
        c._model_path = model_path
        c._user_defined = user_defined

        leader_key = etcd.get_full_path(ETCD_POD_RANK, ETCD_POD_LEADER)
        state_key = etcd.get_full_path(ETCD_STATE, name)

        etcd = etcd._etcd
        status, _ = etcd.transaction(
            compare=[etcd.transactions.value(leader_key) == pod_id, ],
            success=[etcd.transactions.put(state_key, c.to_json()), ],
            failure=[])

        message = "pod_id:{} save_data_checkpoint status:{}".format(pod_id,
                                                                    status)

        if not status:
            raise EdlPutError(message)

        return status


class PaddleState(State):
    def __init__(self,
                 total_batch_size,
                 user_defined=None,
                 optimizer=None,
                 exe=None,
                 program=None):
        super(PaddleState, self).__init__(
            total_batch_size=total_batch_size, user_defined=user_defined)
        self._exe = None
        self._program = None
