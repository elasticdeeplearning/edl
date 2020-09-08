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

import time
from . import data_server_pb2 as pb
from . import data_server_pb2_grpc as pb_grpc
from .client import Client
from .cluster import Cluster
from .exceptions import deserialize_exception, EdlBarrierError
from .utils import logger
from .etcd_db import get_global_etcd


class Conn(object):
    def __init__(self, channel, stub):
        self._channel = channel
        self._stub = stub


class DataServerClient(object):
    def __init__(self):
        self._conn = {}  #endpoint=>(channel, stub)

    def _connect(self, endpoint):
        return channel, stub

    def connect(self, endpoint):
        if endpoint not in self._conn:
            c = grpc.insecure_channel(endpoint)
            s = pb_grpc.DataServerServerStub(channel)
            self._conn[endpoint] = Conn(c, s)

        return self._conn[endpoint]

    @handle_errors_until_timeout
    def get_file_list(self, endpoint, reader_name, pod_id, timeout=6):
        self._connect(endpoint)

        req = pb.FileListRequest()
        req.pod_id = pod_id
        req.reader_name = reader_name

        res = s.GetFileList(req)
        if res.status.type != "":
            deserialize_exception(res.status)

        ret = []
        for m in res.metas:
            ret.append((m.idx, m.path))

        logger.debug("pod client get file_list:{}".format(ret))
        return ret

    @handle_timeout_errors
    def get_batch_data_idx(self,
                           reader_name=None,
                           pod_id=None,
                           endpoint=None,
                           batch_data_ids=None,
                           timeout=6):
        self._connect(endpoint)

        req = pb.BatchDataMeta()
        req.reader_name = reader_name
        req.producer_pod_id = pod_id
        req.data_server_endpoint = endpoint
        for idx in batch_data_ids:
            req.batch_data_ids.append(idx)

        res = s.GetBatchDataMeta(req)
        if res.status.type != "":
            deserialize_exception(res.status)

        for m in res.metas:
            logger.debug("pod client get batch_idx endpoint:{} ids:{}".format(
                m.data_server_endpoint, [x for x in m.batch_data_ids]))
        return res.metas
