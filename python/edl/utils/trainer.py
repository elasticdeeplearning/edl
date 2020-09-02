# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# NOTE: 
# This file is copied from paddle/python/distributed/utils.py
# remove it when paddle's is ready.

import functools
import logging
import time
import os
import signal
import copy
import sys
import subprocess
import uuid
import json
import collections
from .exceptions import *
from . import utils
from enum import IntEnum
from .utils import logger
from .global_vars import TrainStatus


class Trainer(object):
    def __init__(self):
        self._id = None
        self._rank_in_pod = None
        self._gpus = []
        self._endpoint = None
        self._global_rank = None

    def to_json(self):
        d = {
            "id": self._id,
            "rank_in_pod": self._rank_in_pod,
            "gpus": self._gpus,
            "endpoint": self._endpoint,
            "global_rank": self._global_rank,
        }

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)

        self._id = d["id"]
        self._rank_in_pod = d["rank_in_pod"]
        self._gpus = d["gpus"]
        self._endpoint = d["endpoint"]
        self._global_rank = d["global_rank"]

    def __str__(self):
        s = "id:{} rank_in_pod:{} gpus:{} endpoint:{} global_rank:{}".format(
            self._ids, self._rank_in_pod, self._gpus, self._endpoint,
            self._global_rank)

        return s

    def __eq__(self, t):
        if self._id != self._id:
            return False

        if self._gpus != t._gpus or \
            self._endpoint != t._endpoint or \
            self._rank_in_pod != t._rank_in_pod or \
            self._global_rank != t._global_rank :
            return False

        return True

    def __ne__(self, t):
        return not self == t

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def rank_in_pod(self):
        return self._rank_in_pod

    @property
    def gpus(self):
        return self._gpus

    @property
    def endpoint(self):
        return self._endpoint

    def from_pod(self, endpoint, rank_in_pod, gpus):
        self._id = str(uuid.uuid1())
        self._global_rank = None
        self._rank_in_pod = rank_in_pod
        self._endpoint = endpoint
        self._gpus = gpus

    def from_pb(self, t):
        self._id = t._id
        self._global_rank = t.global_rank
        self._rank_in_pod = t.rank_in_pod
        self._endpoint = t.endpoint
        self._gpus = []
        for g in pod.gpus:
            self.gpus.append(g)
