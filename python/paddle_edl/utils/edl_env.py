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


class JobEnv(object):
    def __init__(self, etcd_endoints=None, job_id=None):
        self.running_env = os.getenv("PADDLE_RUNING_ENV")
        self.job_id = os.getenv("PADDLE_JOB_ID") if job_id is None else job_id
        self.etcd_endpoints = os.getenv(
            "PADDLE_EDL_ETCD_ENPOINTS"
        ) if etcd_endpoints is None else etcd_endoints

        assert self.job_id, "job_id must has valid value "
        assert self.etcd_endpoints, "etcd_endpoints must has valid value "


class PodEnv(object):
    def __init__(self, gpu_num, pod_id=None, trainer_ports=None):
        self.pod_id = os.getenv("PADDLE_POD_ID") if pod_id is None else pod_id
        assert self.pod_id, "pod_id must has valid value "
        ports = os.getenv(
            "PADDLE_TRAINER_PORTS") if trainer_ports is None else trainer_ports
        assert ports, "PADDLE_TRAINER_PORTS must has valid value"
        self.trainer_ports = ports.split(",")
        assert len(self.trainer_ports) == gpu_num, "one gpu one port"


class TrainerEnv(object):
    def __init__(self, trainer_id=None):
        self.trainer_id = os.getenv(
            "PADDLE_TRAINER_ID") if trainer_id is None else trainer_id
        assert self.trainer_id, "trainer_id must has valid value "
