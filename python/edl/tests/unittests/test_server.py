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
from edl.discovery.etcd_client import EtcdClient
import time
import threading
import os
import copy
import atexit
from etcd3.events import PutEvent, DeleteEvent

from edl.utils.edl_env import JobEnv
from edl.collective.launch import _parse_args, _convert_args_to_dict
from edl.utils.pod_server import PodServer
from edl.utils.pod import Pod
from edl.utils.pod_client import PodServerClient
from edl.utils.exceptions import EdlBarrierError
import edl.utils.utils as utils
from edl.utils.global_vars import *
from edl.utils.generate_cluster import GenerateCluster
from edl.utils.etcd_db import get_global_etcd
from edl.utils.leader_register import LeaderRegister
from threading import Thread

g_job_id = "test_barrier"
g_etcd_endpoints = "127.0.0.1:2379"


class TestBarrier(unittest.TestCase):
    def setUp(self):
        utils.get_logger(log_level=10)
        self._etcd = EtcdClient([g_etcd_endpoints], root=g_job_id)
        self._etcd.init()
        self._db = get_global_etcd([g_etcd_endpoints], g_job_id)

        #self._args = _parse_args()
        self._old_environ = copy.copy(dict(os.environ))
        proc_env = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": g_job_id,
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_PATH": "test_register_path",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".test_register_cache",
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0",
            "PADDLE_EDL_NODES_RANGE": "1:4",
            "PADDLE_EDL_NPROC_PERNODE": "1",
            "PADDLE_ETCD_ENDPOINTS": "127.0.0.1:2379",
            "PADDLE_EDLNODES_RANAGE": "2:2",
            "CUDA_VISIBLE_DEVICES": "0",
            "PADDLE_TRAINER_PORTS": "6670"
        }
        os.environ.pop("https_proxy", None)
        os.environ.pop("http_proxy", None)
        os.environ.update(proc_env)

        #args_dict = _convert_args_to_dict(None)
        self._job_env = JobEnv(None)

        self._etcd.remove_service(ETCD_POD_RESOURCE)
        self._etcd.remove_service(ETCD_POD_RANK)
        self._etcd.remove_service(ETCD_POD_STATUS)
        self._etcd.remove_service(ETCD_JOB_STATUS)
        self._etcd.remove_service(ETCD_TRAIN_STATUS)
        self._etcd.remove_service(ETCD_CLUSTER)
        self._etcd.remove_service(ETCD_READER)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)

    def register_pod(self, job_env):
        pod = Pod()
        pod.from_env(job_env)
        s = PodServer(self._job_env, pod)
        s.start()
        self._etcd.set_server_permanent(ETCD_POD_RESOURCE,
                                        pod.get_id(), pod.to_json())
        self._etcd.set_server_permanent(ETCD_POD_STATUS,
                                        pod.get_id(), pod.to_json())
        print("set permanent:", self._etcd.get_full_path(ETCD_POD_RESOURCE,
                                                         pod.get_id()))

        self._db.set_pod_status(pod.get_id(), Status.INITIAL)
        print("set permanent:", self._etcd.get_full_path(ETCD_POD_STATUS,
                                                         pod.get_id()))

        return pod, s

    def _test_server(self):
        pod_0, server_0 = self.register_pod(self._job_env)
        self._etcd.set_server_permanent(ETCD_POD_RANK, ETCD_POD_LEADER,
                                        pod_0.get_id())
        print("set permanent:", self._etcd.get_full_path(ETCD_POD_RANK,
                                                         ETCD_POD_LEADER))

        pod_1, server_1 = self.register_pod(self._job_env)

        generater = GenerateCluster(self._job_env, pod_0.get_id())
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
            generater.stop()

        try:
            cluster_1 = c.barrier(
                self._job_env.job_id, pod_1.get_id(), timeout=15)
        except:
            generater.stop()

        self.assertNotEqual(cluster_1, None)
        generater.stop()


if __name__ == '__main__':
    unittest.main()
