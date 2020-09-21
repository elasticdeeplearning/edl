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
from ..discovery.etcd_client import EtcdClient

g_etcd = None


def get_global_etcd(etcd_endpoints=None, job_id=None):
    global g_etcd
    if g_etcd is None:
        assert etcd_endpoints is not None and job_id is not None
        g_etcd = EtcdClient(endpoints=etcd_endpoints, root=job_id, timeout=6)
        g_etcd.init()
        return g_etcd

    return g_etcd
