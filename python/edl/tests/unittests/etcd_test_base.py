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

import copy
import edl.utils.constants as constants
import edl.utils.log_utils as log_utils
import os
import unittest
from edl.discovery.etcd_client import EtcdClient
from edl.utils import env as edl_env

g_etcd_endpoints = "127.0.0.1:2379"


class EtcdTestBase(unittest.TestCase):
    def setUp(self, job_id):
        log_utils.get_logger(log_level=10)
        self._etcd = EtcdClient([g_etcd_endpoints], root=job_id)
        self._etcd.init()

        self._old_environ = copy.copy(dict(os.environ))
        proc_env = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": job_id,
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

        self._job_env = edl_env.JobEnv(None)
        constants.clean_etcd(self._etcd)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)
        constants.clean_etcd(self._etcd)
