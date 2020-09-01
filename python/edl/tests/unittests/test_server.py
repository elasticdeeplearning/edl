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
from etcd3.events import PutEvent, DeleteEvent

from edl.utils.edl_env import JobEnv
from edl.collective.launch import _parse_args, _convert_args_to_dict
from edl.utils.pod_server import PodServer
from edl.utils.pod import Pod
from edl.utils.pod_client import PodServerClient
from edl.utils.exceptions import EdlBarrierError
import edl.utils.utils as utils


class TestServer(unittest.TestCase):
    def setUp(self):
        utils.get_logger(log_level=10)
        self.etcd = EtcdClient()
        self.etcd.init()

        self._args = _parse_args()
        self._old_environ = copy.copy(dict(os.environ))
        proc_env = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_server",
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

        args_dict = _convert_args_to_dict(self._args)
        self._job_env = JobEnv(args_dict)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)

    def test_server(self):
        pod = Pod()
        pod.from_env(self._job_env)
        s = PodServer(self._job_env, pod)
        s.start()
        time.sleep(5)
        try:
            c = PodServerClient(pod.endpoint)
            cluster = c.barrier(self._job_env.job_id, pod.get_id(), timeout=0)
            self.assertEqual(cluster, None)
        except EdlBarrierError as e:
            pass
