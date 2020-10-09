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
from edl.utils import cluster as edl_cluster


class TestCluster(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestCluster, self).setUp("test_cluster")

    def test_cluster_basic(self):
        cluster = edl_cluster.Cluster()

        cluster2 = edl_cluster.Cluster()
        cluster2.from_json(cluster.to_json())
        self.assertEqual(cluster, cluster2)


if __name__ == "__main__":
    unittest.main()
