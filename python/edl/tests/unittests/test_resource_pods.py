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
from edl.utils import constants
from edl.utils import resource_pods


class TestRegister(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestRegister, self).setUp("test_register")

    def test_register_resource_pod(self):
        ttl = constants.ETCD_TTL
        register1 = resource_pods.PodResourceRegister(ttl)
        register2 = resource_pods.PodResourceRegister(ttl)
        pods_dict=resource_pods.get_resource_pods_dict()
        self.assertEqual(len(pods_dict),2)

        register1.stop()
        register2.stop()
        time.sleep(ttl+1)
        pods_dict=resource_pods.get_resource_pods_dict()
        self.assertEqual(len(pods_dict),0)
