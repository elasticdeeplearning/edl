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
from edl.utils import cluster as edl_cluster
from edl.utils import constants


class TestWatcher(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestWatcher, self).setUp("test_watcher")

    def test_watcher_stage_changed(self):
        cluster = edl_cluster.Cluster()
        cluster.stage = "0"
        etcd.set_server_permanent(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json)
        watcher = cluster_watcher.Watcher(self._job_env, cluster)

        cluster.statage = "1"
        etcd.set_server_permanent(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json)
        time.sleep(constants.ETCD_TTL)
        self.assertTrue(watcher.changed)

    def test_watch_valid(self):
        try:
            cluster = edl_cluster.Cluster()
            etcd.set_server_permanent(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json)
            watcher = cluster_watcher.Watcher(self._job_env, cluster)
            etcd.remove_server(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER)
            time.sleep(constants.ETCD_TTL)
        except exceptions.EdlTableError as e:
            pass

    def test_watcher_ids_changed(self):
        cluster = edl_cluster.Cluster()
        etcd.set_server_permanent(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json)
        watcher = cluster_watcher.Watcher(self._job_env, cluster)

        pod=edl_pod.Pod()
        cluster._pods.append(pod)
        etcd.set_server_permanent(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json)
        time.sleep(constants.ETCD_TTL)
        self.assertTrue(watcher.changed)

if __name__ == '__main__':
    unittest.main()
