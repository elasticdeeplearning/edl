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
import copy
import json
import six
import uuid

from . import constants
from . import exceptions
from .pod import Pod


class Cluster(object):
    def __init__(self):
        self._pods = []
        self._stage = None
        self._status = constants.Status.INITIAL

    def __str__(self):
        return "pods:{} job_stage:{} status:{}".format(
            [str(pod) for pod in self._pods], self._stage, self._status)

    def details(self):
        return "pods:{} job_stage:{} status:{}".format(
            [pod.details() for pod in self._pods], self._stage, self._status)

    def __eq__(self, cluster):
        if cluster is None:
            return False

        if self._stage != cluster._stage:
            return False

        if len(self._pods) != len(cluster.pods):
            return False

        for a, b in zip(self._pods, cluster.pods):
            if a != b:
                return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self._pods = copy.copy(cluster.pods)

    def get_trainers_nranks(self):
        return len(self.get_trainers_endpoints())

    def get_pods_nranks(self):
        return len(self._pods)

    def get_pods(self):
        return self._pods

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

    def from_pb(self, cluster):
        self._stage = cluster._stage
        self._pods = []
        for pod in cluster:
            p = Pod()
            p.from_pb(pod)
            self._pods.append(p)

    # FIXME(gongwb): use from_pb, to_pb later
    def to_json(self):
        d = {}
        for i, pod in enumerate(self._pods):
            d[i] = pod.to_json()

        d = {"pods": d, "stage": self._stage, "status": int(self._status)}

        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)
        pods = d["pods"]
        self._stage = d["stage"]
        self._status = d["status"]

        od = collections.OrderedDict(sorted(d["pods"].items()))
        pods = []
        for i, (key, value) in enumerate(six.iteritems(od)):
            pod = Pod()
            if i != int(key):
                raise exceptions.EdlRankError(
                    "rank:{} is not exists in {}".format(i, d))
            pod.from_json(value)
            pods.append(pod)

        self._pods = pods

    def get_leader_endpoint(self):
        assert len(self._pods) > 0
        return "{}:{}".format(self._pods[0].addr, self._pods[0].port)

    def get_leader_id(self):
        assert len(self._pods) > 0
        return self._pods[0].get_id()

    def new_stage(self):
        self._stage = str(uuid.uuid1())

    @property
    def stage(self):
        return self._stage

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        self._status = s
