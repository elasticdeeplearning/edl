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

from paddle_edl.utils.utils import get_extern_ip, logger


class JobEnv(object):
    def __init__(self, etcd_endoints=None, job_id=None):
        self.running_env = os.getenv("PADDLE_RUNING_ENV")
        self.job_id = os.getenv("PADDLE_JOB_ID") if job_id is None else job_id
        self.etcd_endpoints = os.getenv(
            "PADDLE_ETCD_ENPOINTS") if etcd_endpoints is None else etcd_endoints

        assert self.job_id, "job_id must has valid value "
        assert self.etcd_endpoints, "etcd_endpoints must has valid value "


class PodEnv(object):
    def __init__(self,
                 gpu_num,
                 pod_id=None,
                 pod_ip=None,
                 pod_port=None,
                 trainer_ports=None):
        self.pod_id = os.getenv("PADDLE_POD_ID") if pod_id is None else pod_id
        self.pod_port = os.getenv(
            "PADDLE_POD_PORT") if pod_port is None else pod_port
        self.pod_addr = os.getenv(
            "PADDLE_POD_IP") if pod_ip is None else pod_ip
        assert self.pod_id, "pod_id must has valid value "
        assert self.pod_port, "pod_port must has valid value "
        assert self.pod_addr, "pod_ip must has valid value "
        self.pod_endpoint = "{}:{}".format(self.pod_addr, self.pod_port)

        ports = os.getenv(
            "PADDLE_TRAINER_PORTS") if trainer_ports is None else trainer_ports
        assert ports, "PADDLE_TRAINER_PORTS must has valid value"
        self.trainer_ports = ports.split(",")
        assert len(self.trainer_ports) == gpu_num, "one gpu one port"


class TrainerEnv(object):
    def __init__(self, trainer_id=None):
        self.trainer_rank_in_pod = os.getenv(
            "PADDLE_TRAINER_RANK_IN_POD") if trainer_id is None else trainer_id
        self.trainer_global_rank = os.getenv(
            "PADDLE_TRAINER_GLOBAL_RANK") if trainer_id is None else trainer_id
        assert self.trainer_rank_in_pod, "trainer_rank_in_pod must has valid value "
        assert self.trainer_global_rank, "global_rank must has valid value "
