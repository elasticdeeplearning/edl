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


class Pod(object):
    def __init__(self):
        self._id = None
        self._rank = None
        self._port = None
        self._trainer_ports = None
        self._addr = None
        self._gpus = None
        self._trainers = None
        self._master = None  # candidates
        self._trainers = None

    def init_from_env(self, job_env):
        # uuid
        self._id = uuid.uuid1()

        # trainer ports
        self._get_ports()

        # gpus
        self._gpus = job_env.selected_gpus

        # hostname, ip
        _, self._addr = utils.get_host_name_ip()

        # init trainers
        self._trainers = []
        n = self.nproc_per_node / len(self._gpus)
        for i in range(job_env.nproc_per_node):
            b = i * n
            e = i * n + n
            if i == job_env.nproc_per_node - 1:
                e = job_env.nproc_per_node

            t = Trainer()
            t.init_from_pod(
                self, endpoint=endpoint, rank_in_pod=i, gpus=self._gpus[b:e])
            self._trainers.append(t)

    def _get_ports(self):
        if self._job_env.run_platform == "PADDLE_CLOUD":
            ports = os.getenv("PADDLE_TRAINER_PORTS", "")
            self._trainer_ports = ports.split(",")

            assert len(ports) >= len(self._gpus), \
                "port num:{} must large than gpus:{}".format(len(self._trainer_ports), len(self._gpus))
            logger.info("get ports from env:{}".format(self._trainer_ports))
        else:
            self._trainer_ports = utils.find_free_ports(len(_selected_gpus))
            logger.info("get ports from unused:{}".format(self._trainer_ports))

    def init_from_pb(self, pod):
        self._id = pod.id
        self._rank = pod.rank
        self._addr = pod.addr
        self._trainer_ports = pod.trainer_ports
        self._gpus = []
        for g in pod.gpus:
            self._gpus.append(g)

        self.trainers = []
        for trainer in pod.trainers:
            t = Trainer()
            t.init_from_pb(trainer)

            self.trainers.append(t)

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


class Trainer(object):
    def __init__(self):
        self._id = None
        self._rank_in_world = None
        self._gpus = []
        self._endpoint = None
        self._global_rank = None

    def __str__(self):
        return "gpu:{} endpoint:{} rank_in_pod:{} global_rank:{}".format(
            self.gpus, self.endpoint, self.rank_in_pod, self.global_rank)

    def __eq__(self, t):
        if len(self.gpus) != len(t.gpus):
            return False

        if self._endpoint != t.endpoint or \
                self._global_rank != t.global_rank :
            return False

        for a, b in zip(self.gpus, t.gpus):
            if a != b:
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

    def init_from_pod(self, endpoint, rank_in_pod, gpus):
        self._id = uuid.uuid1()
        self._global_rank = None
        self._rank_in_pod = _rank_in_pod
        self._endpoint = endpoint
        self._gpus = gpus

    def init_from_pb(self, t):
        self._id = t.id
        self._global_rank = t.global_rank
        self._rank_in_pod = t.rank_in_pod
        self._endpoint = t.endpoint
        self._gpus = []
        for g in pod.gpus:
            self.gpus.append(g)


class Master(object):
    def __init__(self):
        self._pod_id = None
        self._pod_rank = None
        self._endpoint = None

    def init_from_etcd(self):
        pass


class Cluster(object):
    def __init__(self):
        self.pods = []
        self.job_stage = None

    def __str__(self):
        return "pods:{} job_stage:{}".format([str(pod) for pod in self.pods],
                                             self.job_stage)

    def details(self):
        return "pods:{} job_stage:{}".format(
            [pod.details() for pod in self.pods], self.job_stage)

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
