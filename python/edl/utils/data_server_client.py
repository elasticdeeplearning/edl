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
        self.channel = channel
        self.stub = stub
        self.lock = threading.Lock()


class DataServerClient(object):
    def __init__(self):
        self._conn = {}  #endpoint=>(channel, stub)

    def _connect(self, endpoint):
        return channel, stub

    @handle_errors_until_timeout
    def connect(self, endpoint, timeout=30):
        if endpoint not in self._conn:
            c = grpc.insecure_channel(endpoint)
            s = pb_grpc.DataServerServerStub(channel)
            self._conn[endpoint] = Conn(c, s)

        return self._conn[endpoint]

    @handle_errors_until_timeout
    def get_file_list(self,
                      leader_endpoint,
                      reader_name,
                      pod_id,
                      file_list,
                      timeout=30):
        conn = self.connect(leader_endpoint)

        req = pb.FileListRequest()
        req.pod_id = pod_id
        req.reader_name = reader_name
        for l in file_list:
            req.file_list.append(l)

        with conn.lock:
            res = conn.stub.GetFileList(req)
        if res.status.type != "":
            deserialize_exception(res.status)

        ret = []
        for m in res.file_list:
            ret.append((m.idx, m.path))

        logger.debug("pod client get file_list:{}".format(ret))
        return ret

    @handle_errors_until_timeout
    def balance_batch_data(self,
                           reader_leader_endpoint,
                           reader_name,
                           pod_id,
                           dataserver_endpoint,
                           batch_data_ids=None,
                           timeout=30):
        conn = self.connect(reader_leader_endpoint)

        req = pb.BatchDataRequest()
        req.reader_name = reader_name
        req.producer_pod_id = pod_id
        req.consumer_pod_id = None
        req.data_server_endpoint = endpoint
        for i in batch_data_ids:
            b = pb.BatchData()
            b.batch_data_id = i
            req.data.append(b)

        with conn.lock:
            res = conn.stub.GetBatchData(req)

        if res.status.type != "":
            deserialize_exception(res.status)

        logger.debug("pod client get batch_data meta:{}".format(
            batch_data_response_to_string(res)))
        return res.ret

    @handle_errors_until_timeout
    def get_batch_data(self, req, time=30):
        """
        return BatchDataResponse
        """
        conn = self.connect(reader_leader_endpoint)

        with conn.lock:
            res = conn.stub.GetBatchData(req)
        if res.status.type != "":
            deserialize_exception(res.status)

        logger.debug("pod client get batch_data meta:{}".format(
            batch_data_response_to_string(res)))
        return res.ret
