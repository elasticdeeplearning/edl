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
import six
import time
import unittest
from edl.tests.unittests import etcd_test_base
from edl.utils import constants
from edl.utils import pod as edl_pod
from edl.utils import resource_pods


class TestRegister(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestRegister, self).setUp("test_register")

    def test_register_resource_pod(self):
        try:
            pod0 = edl_pod.Pod()
            pod0._id = "0"

            pod1 = edl_pod.Pod()
            pod1._id = "1"

            ttl = constants.ETCD_TTL
            register1 = resource_pods.Register(
                self._job_env, pod_id="0", pod_json=pod0.to_json(), ttl=ttl
            )
            register2 = resource_pods.Register(
                self._job_env, pod_id="1", pod_json=pod1.to_json(), ttl=ttl
            )

            # check if the ttl is valid
            time.sleep(ttl + 2)

            pods = resource_pods.load_from_etcd(self._etcd, timeout=15)
            self.assertEqual(len(pods), 2)
            for pod_id, pod in six.iteritems(pods):
                if pod_id == "0":
                    self.assertEqual(pod, pod0)
                elif pod_id == "1":
                    self.assertEqual(pod, pod1)
                else:
                    raise Exception("not supported pod_id:{}".format(pod_id))
        except Exception as e:
            raise e
        finally:
            register1.stop()
            register2.stop()

        time.sleep(ttl + 1)
        pods_dict = resource_pods.load_from_etcd(self._etcd, timeout=15)
        self.assertEqual(len(pods_dict), 0)


if __name__ == "__main__":
    unittest.main()
