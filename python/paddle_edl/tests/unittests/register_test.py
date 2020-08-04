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
from paddle_edl.discovery.etcd_client import EtcdClient
import time
import threading
from etcd3.events import PutEvent, DeleteEvent

from paddle_edl.utils.utils import bytes_to_string
from paddle_edl.utils.register import PodRegister, MasterRegister
from paddle_edl.utils.cluster import Pod, Trainer
from paddle_edl.utils.edl_env import JobEnv
from paddle_edl.utils.edl_launch import _parse_args


class TestRegister(unittest.TestCase):
    def setUp(self):
        self.etcd = EtcdClient()
        self.etcd.init()

        self._pod_register = None
        self._master_register = None
        self._args = _parse_args()
        self._old_environ = dict(os.environ)
        proc_env = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_register",
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_PATH": "test_register_path",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".test_register_cache",
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0",
            "PADDLE_EDL_NODES_RANGE": "1:4",
            "PADDLE_EDL_NPROC_PERNODE": "1"
        }
        os.environ.update(proc_env)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)
        self._pod_register.stop()
        self._master_register.stop()

    def test_register_pod(self):
        pod = Pod()
        pod.init_from_env()

        job_env = JobEnv(self._args)
        pod_env = Pod(job_env)

        self._pod_register = PodRegister(job_env, pod_env)
        original = pod_env.to_json()

        servers = self.etcd.get_service("pod")
        assert len(servers) == 1, "key must not alive when expired."
        s = servers[0]
        assert s.info == original

    def test_register_master(self):
        pod = Pod()
        pod.init_from_env()

        job_env = JobEnv(self._args)
        pod_env = Pod(job_env)
        self._master_register = MasterRegister(job_env, pod)

        original = pod_env.to_json()

        servers = self.etcd.get_service("pod")
        assert len(servers) == 1, "key must not alive when expired."
        s = servers[0]
        assert s.info == original
