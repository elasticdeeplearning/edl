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

from edl.utils import constants


def get_train_status_table_key(self, server_name):
    return self._etcd.get_full_path(constants.ETCD_TRAIN_STATUS, server_name)


def get_cluster_table_key(self):
    return self._etcd.get_full_path(constants.ETCD_CLUSTER,
                                    constants.ETCD_CLUSTER)


def get_rank_table_key(self):
    return self._etcd.get_full_path(constants.ETCD_POD_RANK,
                                    constants.ETCD_POD_LEADER)
