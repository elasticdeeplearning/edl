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
from edl.utils import cluster as edl_cluster
from edl.utils import etcd_utils
from edl.utils import exceptions
from edl.utils import string_utils


def get_pod_leader_id(etcd):
    value = etcd.get_value(constants.ETCD_POD_RANK, constants.ETCD_POD_LEADER)
    if value is None:
        return None

    return string_utils.bytes_to_string(value)


def get_pod_leader(etcd):
    leader_id = get_pod_leader_id(etcd)
    cluster = edl_cluster.load_from_etcd(etcd)

    if leader_id is None:
        raise exceptions.EdlTableError("leader_id={}:{}".format(
            etcd_utils.get_rank_table_key(), leader_id))

    if cluster is None:
        raise exceptions.EdlTableError("cluster={}:{}".format(
            etcd_utils.get_cluster_table_key(), cluster))

    if cluster.pods[0].get_id() != leader_id:
        raise exceptions.EdlLeaderError("{} not equal to {}".format(
            cluster.pods[0].get_id(), leader_id))

    return cluster.pods[0]

