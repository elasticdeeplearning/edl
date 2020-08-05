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
import socket
import time
import os
import signal
import copy
import sys
import subprocess
from contextlib import closing
import socket
import uuid
import collections
from .exceptions import *


class PodStatus(Enum):
    INITIAL = 0
    RUNNING = 1
    PENDING = 2
    COMPLETE = 3


class Pod(object):
    def __init__(self):
        self._id = None  # id is not rank even they has same value
        self._rank = None
        self._trainer_ports = None
        self._addr = None
        self._gpus = None
        self._trainers = None
        self._port = None
        self._status = PODStatus.INITIAL  # status maybe changed

    def to_json(self):
        d = {
            "id": self._id,
            "rank": self._rank,
            "port": self._port,
            "trainer_ports": self._trainer_ports,
            "addr": self._addr,
            "gpus": self._gpus,
        }

        d["trainers"] = {}
        for i, t in enumerate(trainers):
            d["trainers"][i] = t.to_json()

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)

        self._id = d["id"]
        self._rank = d["rank"]
        self._addr = pod.addr
        self._port = self._job_env.pod_port
        self._trainer_ports = pod.trainer_ports
        self._gpus = d["gpus"]

        self.trainers = []

        od = collections.OrderedDict(sorted(d["trainers"].items()))
        for i, (key, value) in enumerate(od.iteritems()):
            t = Trainer()
            t.from_json(value)

            self._trainers.append(t)

    def from_env(self, job_env):
        self._job_env = job_env

        # uuid
        self._id = uuid.uuid1()
        self._trainer_ports = job_env.trainer_ports

        # gpus
        self._gpus = job_env.selected_gpus

        # hostname, ip
        _, self._addr = utils.get_host_name_ip()

        # init trainers
        self._trainers = []
        n = self.nproc_per_node / len(self._gpus)
        assert n>=1, \
            "self.nproc_per_node:{} / len(self._gpus):{} must large than 1".format(self.nproc_per_node,len(self._gpus))

        for i in range(job_env.nproc_per_node):
            b = i * n
            e = i * n + n
            if i == job_env.nproc_per_node - 1:
                e = job_env.nproc_per_node

            endpoint = "{}:{}".format(self._addr, self._trainer_port[i])

            t = Trainer()
            t.from_pod(
                self, endpoint=endpoint, rank_in_pod=i, gpus=self._gpus[b:e])
            self._trainers.append(t)

    def _start_sever(self):
        return port

    def from_pb(self, pod):
        self._id = pod.id
        self._rank = pod.rank
        self._addr = pod.addr
        self._port = pod.port
        self._trainer_ports = pod.trainer_ports
        self._gpus = []
        for g in pod.gpus:
            self._gpus.append(g)

        self.trainers = []
        for trainer in pod.trainers:
            t = Trainer()
            t.from_pb(trainer)

            self.trainers.append(t)

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} gpus:{} status:{} trainers_num:{}".format(
            self._rank, self._id, self._addr, self._port, self._gpus,
            self._status, len(self._trainers))

    def details(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{} steatus:{} trainers:{}".format(
            self._rank, self._id, self._addr, self._port, self._gpus,
            self._status, [str(t) for t in self.trainers])

    def __eq__(self, pod):
        if self._rank != pod._rank or \
                self._id != pod._id or \
                self._trainer_ports != pod._trainer_ports or \
                self._addr != pod._addr or \
                self._gpus != pod._gpus or \
                self._trainers != pod._trainers or \
                self._port != pod._port:  # not with status
            logger.debug("pod {} != pod".format(self, pod))
            return False

        if len(self.trainers) != len(pod.trainers):
            logger.debug("trainers {} != {}".format(self.trainers,
                                                    pod.trainers))
            return False

        for i in range(len(self.trainers)):
            if self.trainers[i] != pod.trainers[i]:
                logger.debug("trainer {} != {}".format(self.trainers[i],
                                                       pod.trainers[i]))
                return False

        return True

    def __ne__(self, pod):
        return not self == pod

    @property
    def gpus(self):
        return self._gpus

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value
        for i, t in enumerate(self._tainers()):
            self._rank_in_pod = i
            t._global_rank = self._rank + self._rank_in_pod

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, v):
        self._port = port

    @property
    def addr(self):
        return self._addr


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
        return "id:{} rank_in_pod:{} gpus:{} endpoint:{} global_rank:{}".format(
            self._ids, self._rank_in_pod, self._gpus, self._endpoint,
            self._global_rank)

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
        return self._global_rank

    def from_pod(self, endpoint, rank_in_pod, gpus):
        self._id = uuid.uuid1()
        self._global_rank = None
        self._rank_in_pod = _rank_in_pod
        self._endpoint = endpoint
        self._gpus = gpus

    def from_pb(self, t):
        self._id = t.id
        self._global_rank = t.global_rank
        self._rank_in_pod = t.rank_in_pod
        self._endpoint = t.endpoint
        self._gpus = []
        for g in pod.gpus:
            self.gpus.append(g)


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

    def trainers_nranks(self):
        return len(self.trainers_endpoints())

    def pods_nranks(self):
        return len(self._pods)

    def trainers_endpoints(self):
        r = []
        for pod in self._pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def pods_endpoints(self):
        r = []
        for pod in self._pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self._pods:
            if str(pod_id) == str(pod.id):
                return pod

        return None

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

    def from_json(self, d):
        """
        d = {rank:json, ...}
        """
        od = collections.OrderedDict(sorted(d.items()))
        pods = []
        for i, (key, value) in enumerate(old.iteritems()):
            pod = Pod()
            if i != key:
                raise EdlRankError("rank:{} is not exists:{}".format(i, d))
            pods.append(pod.from_json(value))

        self._pods = pods

    def get_master_endpoint(self):
        return "{}:{}".format(self._pods[0].addr, self._pods[0].port)
