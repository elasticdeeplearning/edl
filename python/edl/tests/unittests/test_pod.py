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
import unittest

from edl.tests.unittests import etcd_test_base
from edl.utils import pod as edl_pod


class TestPod(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestPod, self).setUp("test_pod")

    def test_pod(self):
        pod = edl_pod.Pod()
        pod.from_env(self._job_env)

        pod2 = edl_pod.Pod()
        pod2.from_json(pod.to_json())
        self.assertEqual(pod, pod2)


if __name__ == '__main__':
    unittest.main()
