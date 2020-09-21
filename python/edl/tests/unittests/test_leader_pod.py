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

import time
import unittest
from edl.tests.unittests import etcd_test_base
from edl.utils import constants
from edl.utils import leader_pod
from edl.utils import pod as edl_pod
from edl.utils import resource_pods


class TestLeaderPod(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestGenerate, self).setUp("test_leader_pod")

    def _add_pod(self):
        pod = edl_pod.Pod()
        pod.from_env(self._job_env)
        leader_register=leader_pod.Register(self._job_env, pod.pod_id)
        resource_register=resource_pods.Register(self._job_env,
                                                 pod_id=pod.pod_id, pod_json=pod.to_json(), ttl=constants.ETCD_TTL)

        return (pod, leader_register, resource_register)

    def test_seize_leader(self):
        pod0, leader_register0, resource_register0 = self._add_pod()
        time.sleep(constants.ETCD_TTL)
        pod1, leader_register1, resource_register1 = self._add_pod()

        leader_id = leader_pod.get_pod_leader_id()
        self.assertEqual(pod0.pod_id, leader_id)

        leader_register0.stop()
        time.sleep(constants.ETCD_TTL)

        leader_id = leader_pod.get_pod_leader_id()
        self.assertEqual(pod1.pod_id, leader_id)
        leader_register1.stop()

        resource_register0.stop()
        resource_register1.stop()


if __name__ == '__main__':
    unittest.main()