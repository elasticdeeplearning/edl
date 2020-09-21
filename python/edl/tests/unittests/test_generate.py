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

import sys
import unittest
from edl.tests.unittests import etcd_test_base
from edl.utils import cluster as edl_cluster
from edl.utils import constants
from edl.utils import status as edl_status
from edl.utils.exceptions import EdlBarrierError
from edl.utils.pod import Pod
from edl.utils.pod_server import PodServer
from edl.utils.pod_server_client import PodServerClient


class TestGenerate(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestGenerate, self).setUp("test_generate")

    def register_pod(self, job_env):
        pod = Pod()
        pod.from_env(job_env)
        s = PodServer(self._job_env, pod)
        s.start()
        self._etcd.set_server_permanent(constants.ETCD_POD_RESOURCE,
                                        pod.get_id(), pod.to_json())
        self._etcd.set_server_permanent(constants.ETCD_POD_STATUS,
                                        pod.get_id(), pod.to_json())
        print("set permanent:",
              self._etcd.get_full_path(constants.ETCD_POD_RESOURCE,
                                       pod.get_id()))

        edl_status.save_pod_status_to_etcd(self._etcd,
                                           pod.get_id(),
                                           edl_status.Status.INITIAL)
        print("set permanent:", self._etcd.get_full_path(
            constants.ETCD_POD_STATUS, pod.get_id()))

        return pod, s

    def test_server(self):
        pod_0, server_0 = self.register_pod(self._job_env)
        self._etcd.set_server_permanent(constants.ETCD_POD_RANK,
                                        constants.ETCD_POD_LEADER,
                                        pod_0.get_id())
        print("set permanent:", self._etcd.get_full_path(
            constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER))

        pod_1, server_1 = self.register_pod(self._job_env)

        generater = edl_cluster.Generator(self._job_env, pod_0.get_id())
        ret = generater.start()

        cluster_0 = None
        clsuter_1 = None
        try:
            c = PodServerClient(pod_0.endpoint)
            cluster_0 = c.barrier(
                self._job_env.job_id, pod_0.get_id(), timeout=0)

            self.assertNotEqual(cluster_0, None)
        except EdlBarrierError as e:
            pass
        except:
            sys.exit(1)
            generater.stop()

        try:
            cluster_1 = c.barrier(
                self._job_env.job_id, pod_1.get_id(), timeout=15)
        except:
            generater.stop()
            sys.exit(1)

        generater.stop()
        self.assertNotEqual(cluster_1, None)


if __name__ == '__main__':
    unittest.main()
