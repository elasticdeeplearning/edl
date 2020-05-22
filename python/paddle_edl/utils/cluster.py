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


class Hdfs(object):
    def __init__(self):
        self.hdfs_ugi = None
        self.hdfs_name = None
        self.hdfs_path = None

    def is_valid(self):
        return self.hdfs_ugi is not None and \
            self.hdfs_name is not None and \
            self.hdfs_path is not None

    def __str__(self):
        return "hdfs_ugi:{} hdfs_name:{} hdfs_path{}".format(
            self.hdfs_ugi, self.hdfs_name, self.hdfs_path)

    def __eq__(self, n):
        return self.hdfs_ugi == n.hdfs_ugi and \
            self.hdfs_name == n.hdfs_name and \
            self.hdfs_path == n.hdfs_path

    def __ne__(self, n):
        return not self == n


class Cluster(object):
    def __init__(self, hdfs=None):
        # self.master = None
        self.pods = []
        self.hdfs = hdfs
        self.job_stage = None

    def __str__(self):
        return "pods:{} job_stage_flag:{} hdfs:{}".format(
            [str(pod) for pod in self.pods], self.job_stage_flag, self.hdfs)

    def details(self):
        return "pods:{} job_stage_flag:{} hdfs:{}".format(
            [pod.details() for pod in self.pods], self.job_stage_flag,
            self.hdfs)

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return False

        for a, b in zip(self.pods, cluster.pods):
            if a != b:
                return False

        if self.job_stage_flag != cluster.job_stage_flag:
            return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self.pods = copy.copy(cluster.pods)

    def trainers_nranks(self):
        return len(self.trainers_endpoints())

    def pods_nranks(self):
        return len(self.pods)

    def trainers_endpoints(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def pods_endpoints(self):
        r = []
        for pod in self.pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self.pods:
            if str(pod_id) == str(pod.id):
                return pod

        return None

    def init_from_pb(self, cluster):
        self.job_stage = cluster.job_stage
        self.pods = []
        for pod in cluster:
            p = Pod()
            p.init_from_pb(pod)
            self.pods.append(p)

    def get_pods_endpoints(self):
        eps = []
        for p in self.pods:
            eps.append(p.endpoint)

        return eps


class Trainer(object):
    def __init__(self):
        self.gpus = []
        self.endpoint = None
        self.rank_in_pod = None
        self.global_rank = None

    def __str__(self):
        return "gpu:{} endpoint:{} rank_in_pod:{} global_rank:{}".format(
            self.gpus, self.endpoint, self.rank_in_pod, self.global_rank)

    def __eq__(self, t):
        if len(self.gpus) != len(t.gpus):
            return False

        if self.endpoint != t.endpoint or \
                self.global_rank != t.global_rank :
            return False

        for a, b in zip(self.gpus, t.gpus):
            if a != b:
                return False

        return True

    def __ne__(self, t):
        return not self == t

    def rank(self):
        return self.global_rank

    def rank_in_pod(self):
        return self.rank_in_pod

    def init_frim_pb(self, trainer):
        self.rank_in_pod = trainer.rank_in_pod
        self.global_rank = trainer.global_rank
        self.endpoint = trainer.endpoint
        self.gpus = []
        for g in pod.gpus:
            self.gpus.append(g)


class Pod(object):
    def __init__(self):
        self.rank = None  # pod_rank
        self.id = None  # pod_id
        self.addr = None
        self.port = None
        self.trainers = []
        self.gpus = []

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{}".format(
            self.rank, self.id, self.addr, self.port, self.gpus)

    def details(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{} trainers:{}".format(
            self.rank, self.id, self.addr, self.port, self.gpus,
            [str(t) for t in self.trainers])

    def __eq__(self, pod):
        if self.rank != pod.rank or \
                self.id != pod.id or \
                self.addr != pod.addr or \
                self.port != pod.port:
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

    def rank(self):
        return self.rank

    def get_visible_gpus(self):
        r = ""
        for g in self.gpus:
            r += "{},".format(g)

        assert r != "", "this pod {} can't see any gpus".format(self)

        r = r[:-1]
        return r

    def init_frim_pb(self, pod):
        self.id = pod.id
        self.rank = pod.rank
        self.addr = pod.addr
        self.port = pod.port
        self.gpus = []
        for g in pod.gpus:
            self.gpus.append(g)

        self.trainers = []
        for trainer in pod.trainers:
            t = Trrainer()
            t.init_from_pb(trainer)

            self.trainers.append(t)

    @property
    def endpoint(self):
        return "{}:{}".format(self.addr, self.port)
