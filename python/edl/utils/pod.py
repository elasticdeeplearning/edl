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

import collections
import json
import six
import uuid

from edl.utils import network_utils
from edl.utils import status as edl_status
from edl.utils.log_utils import logger
from edl.utils import trainer as edl_trainer


class Pod(object):
    def __init__(self):
        self._id = None  # id is not rank even they has same value
        self._rank = None
        self._trainer_ports = None
        self._addr = None
        self._gpus = None
        self._trainers = None
        self._port = None
        self._status = edl_status.Status.INITIAL  # status maybe changed

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
        for i, t in enumerate(self._trainers):
            d["trainers"][i] = t.to_json()

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)

        self._id = d["id"]
        self._rank = d["rank"]
        self._addr = d["addr"]
        self._port = d["port"]
        self._trainer_ports = d["trainer_ports"]
        self._gpus = d["gpus"]

        self._trainers = []

        od = collections.OrderedDict(sorted(d["trainers"].items()))
        for i, (key, value) in enumerate(six.iteritems(od)):
            t = edl_trainer.Trainer()
            t.from_json(value)

            self._trainers.append(t)

    def from_env(self, job_env):
        self._job_env = job_env

        # uuid
        self._id = str(uuid.uuid1())
        self._trainer_ports = job_env.trainer_ports

        # gpus
        self._gpus = job_env.gpus

        # hostname, ip
        _, self._addr = network_utils.get_host_name_ip()

        # init trainers
        self._trainers = []
        n = int(job_env.nproc_per_node / len(job_env.gpus))
        assert n>=1, \
            "self.nproc_per_node:{} / len(self._gpus):{} must large than 1".format(job_env.nproc_per_node,len(job_env.gpus))

        for i in range(job_env.nproc_per_node):
            b = i * n
            e = i * n + n
            if i == job_env.nproc_per_node - 1:
                e = job_env.nproc_per_node

            logger.debug("[b:e]=[{}:{}]".format(b, e))
            endpoint = "{}:{}".format(self._addr, job_env.trainer_ports[i])

            t = edl_trainer.Trainer()
            t.from_pod(
                endpoint=endpoint, rank_in_pod=i, gpus=job_env.gpus[b:e])
            self._trainers.append(t)

    def from_pb(self, pod):
        self._id = pod._id
        self._rank = pod.rank
        self._addr = pod.addr
        self._port = pod.port
        self._trainer_ports = pod.trainer_ports
        self._gpus = []
        for g in pod.gpus:
            self._gpus.append(g)

        self._trainers = []
        for trainer in pod._trainers:
            t = edl_trainer.Trainer()
            t.from_pb(trainer)

            self._trainers.append(t)

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} gpus:{} status:{} trainers_num:{}".format(
            self._rank, self._id, self._addr, self._port, self._gpus,
            self._status, len(self._trainers))

    def details(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{} status:{} trainers:{}".format(
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

        if len(self._trainers) != len(pod._trainers):
            logger.debug("trainers {} != {}".format(self._trainers,
                                                    pod._trainers))
            return False

        for i in range(len(self._trainers)):
            if self._trainers[i] != pod._trainers[i]:
                logger.debug("trainer {} != {}".format(self._trainers[i],
                                                       pod._trainers[i]))
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
        for i, t in enumerate(self._trainers):
            self._rank_in_pod = i
            t._global_rank = self._rank + self._rank_in_pod

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def addr(self):
        return self._addr

    @property
    def endpoint(self):
        return "{}:{}".format(self._addr, self._port)

    @property
    def trainers(self):
        return self._trainers

    @property
    def trainers_num(self):
        return len(self._trainers)

    def get_id(self):
        return self._id
