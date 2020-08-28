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

from edl.discovery.etcd_client import EtcdClient
from edl.utils.global_vars import *
import os

job_id = os.environ["PADDLE_JOB_ID"]
etcd_endpoints = os.environ["PADDLE_ETCD_ENDPOINTS"]

etcd, _ = get_global_etcd([etcd_endpoints], job_id)
etcd.remove_service(ETCD_POD_RESOURCE)
etcd.remove_service(ETCD_POD_RANK)
etcd.remove_service(ETCD_POD_STATUS)
etcd.remove_service(ETCD_JOB_STATUS)
etcd.remove_service(ETCD_TRAIN_STATUS)
etcd.remove_service(ETCD_CLUSTER)
etcd.remove_service(ETCD_READER)
