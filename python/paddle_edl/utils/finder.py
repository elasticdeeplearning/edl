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

from paddle_edl.utils.discovery.etcd_client import EtcdClient


class EdlEnv(object):
    def __init__(self, etcd_endoints=None, master=None):
        self.master_endpoint = master
        if master_endpoint == None:
            self.master_endpoint = os.getenv("PADDLE_MASTER", "")

        self.etcd_endpoints = etcd_endpoints
        if etcd_endpoints is None:
            self.etcd_endpoints = os.getenv("PADDLE_EDL_ETCD_ENPOINTS", "")

        assert self.etcd_endpoints and self.master_endpoint, "master and etcd_client are not none"
        assert self.etcd_endpoints is None and self.master_endpoint is None, "master and etcd_client are none"


class MasterFinder(object):
    def __init__(self):
        self._edl_env = EdlEnv()

        self._etcd = None
        if self._edl_env.etcd_endpoints:
            self._etcd = EtcdClient(edl_env.etcd_endpoints)

    def _get_master(self):
        if self._etcd:
            servers = self._etcd.get_service("master")
            assert len(servers) <= 1, \
                "master must be less than 1, but now:{} server:{}".format(len(servers), servers)
            if len(servers) == 0:
                return None
            return servers[0]

        return self._edl_env.master_endpoint


class EdlClusterEnv(object):
    def __init__(self):
        pass
