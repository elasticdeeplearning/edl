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

import threading
from ..discovery.etcd_client import EtcdClient

ETCD_POD_RESOURCE = "pod_resource"
ETCD_POD_RANK = "pod_rank"
ETCD_POD_STATUS = "pod_status"
ETCD_JOB_STATUS = "job_status"

g_etcd = None
g_etcd_lock = None


def get_global_etcd(etcd_endpoints=None, job_id=None):
    global g_etcd
    global g_etcd_lock

    if g_etcd_lock is None:
        g_etcd_lock = threading.Lock()

    if g_etcd is None:
        g_etcd = EtcdClient(endpoints=etcd_endpoints, root=job_id, timeout=6)
        g_etcd.init()

    return g_etcd, g_etcd_lock
