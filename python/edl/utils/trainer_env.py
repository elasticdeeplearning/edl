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

import os
from edl.utils import job_env


class TrainerEnv(object):
    """
    Parse all envs when edl_launch starts a trainer.
    """

    def __init__(self, args=None):
        self._job_id = os.environ["PADDLE_JOB_ID"]
        self._pod_id = os.environ["PADDLE_POD_ID"]
        self._pod_leader_id = os.environ["EDL_POD_LEADER_ID"]
        self._etcd_endpoints = os.environ["PADDLE_ETCD_ENDPOINTS"]

        self._global_rank = int(os.environ["PADDLE_TRAINER_ID"])
        self._rank_in_pod = int(os.environ["PADDLE_TRAINER_RANK_IN_POD"])
        self._trainer_endpoints = os.environ["PADDLE_TRAINER_ENDPOINTS"]
        self._pod_ids = os.environ["EDL_POD_IDS"].split(",")
        self._ce_test = int(os.getenv("PADDLE_EDL_ONLY_FOR_CE_TEST", "0"))
        self._get_hdfs(args)

    def _get_hdfs(self, args):
        # hdfs
        self._hdfs_home = job_env.get_from_dict_or_env(
            args, "hdfs_home", "PADDLE_EDL_HDFS_HOME"
        )
        self._hdfs_name = job_env.get_from_dict_or_env(
            args, "hdfs_name", "PADDLE_EDL_HDFS_NAME"
        )
        self._hdfs_path = job_env.get_from_dict_or_env(
            args, "hdfs_path", "PADDLE_EDL_HDFS_PATH"
        )
        self._hdfs_ugi = job_env.get_from_dict_or_env(
            args, "hdfs_ugi", "PADDLE_EDL_HDFS_UGI"
        )

        # assert hdfs value
        if not self._ce_test:
            assert (
                len(self._hdfs_home) > 3
                and len(self._hdfs_name) > 6
                and len(self._hdfs_ugi) > 3
                and len(self._hdfs_path) > 0
            ), "hdfs environ must set"
        else:
            assert (
                len(self._hdfs_home) > 3 and len(self._hdfs_path) > 0
            ), "hdfs environ must set"

    @property
    def pod_leader_id(self):
        return self._pod_leader_id

    @property
    def pod_ids(self):
        return self._pod_ids

    @property
    def pod_id(self):
        return self._pod_id

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def rank_in_pod(self):
        return self._rank_in_pod

    @property
    def trainer_endpoints(self):
        return self._trainer_endpoints

    @property
    def size(self):
        return len(self._trainer_endpoints)

    @property
    def job_id(self):
        return self._job_id

    @property
    def etcd_endpoints(self):
        return self._etcd_endpoints
