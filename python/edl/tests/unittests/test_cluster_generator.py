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
from edl.utils import constants
from edl.utils import status as edl_status
from edl.utils.exceptions import EdlBarrierError
from edl.utils import pod as edl_pod
from edl.utils import pod_server
from edl.utils import pod_server_client
from edl.utils import cluster_generator


class TestClusterGenerator(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestClusterGenerator, self).setUp("test_cluster_generator")

    def _register_pod(self, job_env):
        pod = edl_pod.Pod()
        pod.from_env(job_env)
        server = pod_server.PodServer(self._job_env, pod)
        server.start()
        self._etcd.set_server_permanent(constants.ETCD_POD_RESOURCE,
                                        pod.get_id(), pod.to_json())
        self._etcd.set_server_permanent(constants.ETCD_POD_STATUS,
                                        pod.get_id(), pod.to_json())
        print("set permanent:",
              self._etcd.get_full_path(constants.ETCD_POD_RESOURCE,
                                       pod.get_id()))

        edl_status.save_pod_status_to_etcd(
            self._etcd, pod.get_id(), edl_status.Status.INITIAL, timeout=15)
        print("set permanent:", self._etcd.get_full_path(
            constants.ETCD_POD_STATUS, pod.get_id()))

        return pod, server

    def test_barrier(self):
        pod_0, server_0 = self._register_pod(self._job_env)
        self._etcd.set_server_permanent(constants.ETCD_POD_RANK,
                                        constants.ETCD_POD_LEADER,
                                        pod_0.get_id())
        print("set permanent:", self._etcd.get_full_path(
            constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER))

        pod_1, server_1 = self._register_pod(self._job_env)

        generater = cluster_generator.Generator(self._job_env, pod_0.get_id())
        generater.start()

        try:
            client = pod_server_client.Client(pod_0.endpoint)
            cluster_0 = client.barrier(
                self._job_env.job_id, pod_0.get_id(), timeout=0)

            self.assertNotEqual(cluster_0, None)
        except EdlBarrierError as e:
            pass
        except:
            sys.exit(1)
        finally:
            generater.stop()

        try:
            cluster_1 = client.barrier(
                self._job_env.job_id, pod_1.get_id(), timeout=15)
        except:
            sys.exit(1)
        finally:
            generater.stop()

        self.assertNotEqual(cluster_1, None)


if __name__ == '__main__':
    unittest.main()
