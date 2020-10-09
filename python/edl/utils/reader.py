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
import json
from edl.utils import cluster as edl_cluster
from edl.utils import constants
from edl.utils import error_utils
from edl.utils import exceptions
from edl.utils.log_utils import logger


class ReaderMeta(object):
    def __init__(self, name, pod_id, data_server_endpoint):
        self.name = name
        self.pod_id = pod_id
        self.endpoint = data_server_endpoint

    def to_json(self):
        d = {
            "name": self.name,
            "pod_id": self.pod_id,
            "endpoint": self.endpoint,
        }
        return json.dumps(d)

    def from_json(self, s):
        d = json.loads(s)
        self.name = d["name"]
        self.pod_id = d["pod_id"]
        self.endpoint = d["endpoint"]

    def __str_(self):
        return self._to_json()


@error_utils.handle_errors_until_timeout
def save_to_etcd(etcd, reader_name, pod_id, data_server_endpoint, timeout=60):
    meta = ReaderMeta(reader_name, pod_id, data_server_endpoint)
    path = constants.ETCD_READER + "/" + reader_name
    etcd.set_server_permanent(path, pod_id, meta.to_json())


@error_utils.handle_errors_until_timeout
def load_from_etcd(self, etcd, reader_name, pod_id, timeout=60):
    path = constants.ETCD_READER + "/" + reader_name
    with self._lock:
        value = etcd.get_value(path, pod_id)

    if value is None:
        raise exceptions.EdlTableError(
            "path:{}".format(etcd.get_full_path(path, pod_id))
        )

    meta = ReaderMeta()
    meta.from_json(value)
    logger.debug("get reader:{}".format(meta))
    return meta


def check_readers(etcd):
    servers = etcd.get_service(constants.ETCD_READER)

    if len(servers) <= 0:
        raise exceptions.EdlTableError(
            "table:{} has no readers".format(constants.ETCD_READER)
        )

    readers = {}
    for s in servers:
        r = ReaderMeta()
        r.from_json(s.value)

        readers[r.key] = r

    cluster = edl_cluster.get_cluster(etcd)
    if cluster is None:
        raise exceptions.EdlTableError(
            "table:{} has no readers".format(constants.ETCD_CLUSTER)
        )

    if cluster.get_pods_ids_set() != set(readers.keys()):
        raise exceptions.EdlTableError(
            "reader_ids:{} != cluster_pod_ids:{}".format(
                readers.keys(), cluster.get_pods_ids_set()
            )
        )

    logger.debug("get readers:{}".format(readers))
    return readers
