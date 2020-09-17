from edl.utils import edl_cluster
from edl.utils import exceptions
from edl.utils import constants
from edl.utils import string_utils
from edl.utils import etcd_utils

def get_pod_leader_id(etcd):
    value = etcd.get_value(constants.ETCD_POD_RANK,
                                     constants.ETCD_POD_LEADER)
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