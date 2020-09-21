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

from edl.utils import constants
from edl.utils import exceptions
from edl.utils import pod as edl_pod
from edl.utils import error_utils
from edl.utils import status as edl_status
from edl.utils import json_serializable


class Cluster(json_serializable.Serializable):
    def __init__(self):
        self._pods = []
        self._stage = None
        self._status = edl_status.Status.INITIAL

    def __str__(self):
        return "pods:{} job_stage:{} status:{}".format(
            [str(pod) for pod in self._pods], self._stage, self._status)

    def details(self):
        return "pods:{} job_stage:{} status:{}".format(
            [pod.details() for pod in self._pods], self._stage, self._status)

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
            pod = edl_pod.Pod()
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


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, timeout=60):
    value = etcd.get_value(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER)

    if value is None:
        return None

    cluster = Cluster()
    cluster.from_json(value)
    return cluster
