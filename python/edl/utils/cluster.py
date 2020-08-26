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
import json
import uuid
import collections
from .exceptions import *
from . import utils
from enum import IntEnum
from .utils import logger
from .pod import Pod
from .trainer import Trainer
from . import pod_server_pb2 as pod_server_pb


class Cluster(object):
    def __init__(self):
        self._pods = []
        self.job_stage = None

    def __str__(self):
        return "pods:{} job_stage:{}".format([str(pod) for pod in self._pods],
                                             self.job_stage)

    def details(self):
        return "pods:{} job_stage:{}".format(
            [pod.details() for pod in self._pods], self.job_stage)

    def __eq__(self, cluster):
        if len(self._pods) != len(cluster.pods):
            return False

        for a, b in zip(self._pods, cluster.pods):
            if a != b:
                return False

        if self.job_stage_flag != cluster.job_stage_flag:
            return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self._pods = copy.copy(cluster.pods)

    def get_trainers_world_size(self):
        return len(self.get_trainers_endpoints())

    def get_pods_nranks(self):
        return len(self._pods)

    @property
    def pods(self):
        return self._pods

    def get_trainers_endpoints(self):
        r = []
        for pod in self._pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def get_pods_endpoints(self):
        r = []
        for pod in self._pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self._pods:
            if str(pod_id) == str(pod._id):
                return pod

        return None

    def get_pods_ids_list(self):
        ids = []
        for pod in self._pods:
            ids.append(pod._id)

        return ids

    def get_pods_ids_set(self):
        ids = set()
        for pod in self._pods:
            ids.add(pod._id)

        return ids

    def changed_pods(self, new_cluster, print_detail=True):
        disappeared = []
        rank_changed = {}
        for old_pod in self._pods:
            new_pod = new_cluster.get_pod_by_id(self, old_pod._id)
            if new_pod is None:
                disappeared.append(old_pod)
                if print_detail:
                    logger.info("disappeared pods:{}".format(old_pod))

            if old_pod.rank != new_pod.rank:
                rank_changed[old_pod] = copy.copy(new_pod)
                if print_detail:
                    logger.info("pods change from {} to {}".format(old_pod,
                                                                   new_pod))

        return disappeared, rank_changed

    def from_pb(self, cluster):
        self.job_stage = cluster.job_stage
        self._pods = []
        for pod in cluster:
            p = Pod()
            p.from_pb(pod)
            self._pods.append(p)

    def get_pods_endpoints(self):
        eps = []
        for p in self._pods:
            eps.append(p.endpoint)

        return eps

    def from_rank_dict(self, d):
        """
        s = {rank:json, ...}
        """
        od = collections.OrderedDict(sorted(d.items()))
        pods = []
        for i, (key, value) in enumerate(od.iteritems()):
            pod = Pod()
            if i != key:
                raise EdlRankError("rank:{} is not exists in {}".format(i, d))
            pod.from_json(value)
            pods.append(pod)

        self._pods = pods

    # FIXME(gongwb)
    def to_pb_response(self, res):
        assert len(res.pods) == 0
        pb_pod = pod_server_pb.Pod()
        for i, pod in enumerate(self._pods):
            pb_pod.rank = i
            pb_pod.json = pod.to_json()
            res.pods.append(pb_pod)

        return res

    def from_pb_response(self, res):
        d = {}
        for pb_pod in res.pods:
            d[pb_pod.rank] = pb_pod.json

        self.from_rank_dict(d)

    def to_json(self):
        d = {}
        for i, pod in enumerate(self._pods):
            d[i] = pod.to_json()

        return json.dumps(d)

    def get_master_endpoint(self):
        return "{}:{}".format(self._pods[0].addr, self._pods[0].port)


class Leader(object):
    def __init__(self):
        self._cluter = None
        self._pod_id = None
        self._stage = None

    def to_json(self):
        pass

    def from_json(self):
        pass
