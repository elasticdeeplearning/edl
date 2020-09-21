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

import json
import uuid
from edl.utils import json_serializable


class Trainer(json_serializable.Serializable):
    def __init__(self):
        self._id = None
        self._rank_in_pod = None
        self._gpus = []
        self._endpoint = None
        self._global_rank = None

    def __str__(self):
        s = "id:{} rank_in_pod:{} gpus:{} endpoint:{} global_rank:{}".format(
            self._ids, self._rank_in_pod, self._gpus, self._endpoint,
            self._global_rank)

        return s

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
