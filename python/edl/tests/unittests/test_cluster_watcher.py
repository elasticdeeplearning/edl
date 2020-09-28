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
from edl.utils import exceptions
from edl.utils import cluster_watcher
from edl.utils import pod as edl_pod


class TestWatcher(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestWatcher, self).setUp("test_watcher")

    def test_watcher_stage_changed(self):
        cluster = edl_cluster.Cluster()
        cluster._stage = "0"
        print("cluster 0 ids:", cluster.to_json(), cluster.get_pods_ids_list())
        self._etcd.set_server_permanent(
            constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json()
        )
        watcher = cluster_watcher.Watcher(self._job_env, cluster)

        cluster._stage = "1"
        print("cluster 1 ids:", cluster.to_json(), cluster.get_pods_ids_list())
        self._etcd.set_server_permanent(
            constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json()
        )
        time.sleep(constants.ETCD_TTL)
        self.assertTrue(watcher.changed)

    def test_watch_valid(self):
        try:
            cluster = edl_cluster.Cluster()
            self._etcd.set_server_permanent(
                constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json()
            )
            cluster_watcher.Watcher(self._job_env, cluster)
            self._etcd.remove_server(constants.ETCD_CLUSTER, constants.ETCD_CLUSTER)
            time.sleep(constants.ETCD_TTL)
        except exceptions.EdlTableError:
            pass

    def test_watcher_ids_changed(self):
        cluster = edl_cluster.Cluster()
        print("cluster 0 ids:", cluster.to_json(), cluster.get_pods_ids_list())
        self._etcd.set_server_permanent(
            constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json()
        )
        watcher = cluster_watcher.Watcher(self._job_env, cluster)

        pod = edl_pod.Pod()
        cluster._pods.append(pod)
        print("cluster 1 ids:", cluster.to_json(), cluster.get_pods_ids_list())
        self._etcd.set_server_permanent(
            constants.ETCD_CLUSTER, constants.ETCD_CLUSTER, cluster.to_json()
        )
        time.sleep(constants.ETCD_TTL)
        self.assertTrue(watcher.changed)


if __name__ == "__main__":
    unittest.main()
