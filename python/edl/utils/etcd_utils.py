from edl.utils import constants

def get_train_status_table_key(self, server_name):
    return self._etcd.get_full_path(constants.ETCD_TRAIN_STATUS,
                                    server_name)

def get_cluster_table_key(self):
    return self._etcd.get_full_path(constants.ETCD_CLUSTER,
                                    constants.ETCD_CLUSTER)

def get_rank_table_key(self):
    return self._etcd.get_full_path(constants.ETCD_POD_RANK,
                                    constants.ETCD_POD_LEADER)