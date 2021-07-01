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

import grpc
import threading
from edl.utils import data_server_pb2
from edl.utils import data_server_pb2_grpc
from edl.utils import error_utils
from edl.utils import exceptions
from edl.utils import pb_utils
from edl.utils.log_utils import logger


class Conn(object):
    def __init__(self, channel, stub):
        self.channel = channel
        self.stub = stub
        self.lock = threading.Lock()


# FIXME(gongwb): fix protocal with
# https://medium.com/kuranda-labs-engineering/gracefully-handling-grpc-errors-in-a-go-server-python-client-setup-9805a5464692
class Client(object):
    def __init__(self):
        self._conn = {}  # endpoint=>(channel, stub)

    @error_utils.handle_errors_until_timeout
    def _connect(self, endpoint, timeout=30):
        if endpoint not in self._conn:
            c = grpc.insecure_channel(endpoint)
            s = data_server_pb2_grpc.DataServerStub(c)
            self._conn[endpoint] = Conn(c, s)

        return self._conn[endpoint]

    @error_utils.handle_errors_until_timeout
    def get_file_list(
        self, reader_leader_endpoint, reader_name, pod_id, file_list, timeout=60
    ):
        """
        Return list[(idx,path)...] for pod
        """
        conn = self._connect(reader_leader_endpoint, timeout=30)

        req = data_server_pb2.FileListRequest()
        req.pod_id = pod_id
        req.reader_name = reader_name
        for idx, path in enumerate(file_list):
            ele = data_server_pb2.FileListElement()
            ele.idx = idx
            ele.path = path
            req.file_list.append(ele)

        with conn.lock:
            res = conn.stub.GetFileList(req)
        if res.status.type != "":
            exceptions.deserialize(res.status)

        logger.debug("pod client get file_list:{}".format(res.file_list))
        return res.file_list

    @error_utils.handle_errors_until_timeout
    def report_batch_data_meta(
        self,
        reader_leader_endpoint,
        reader_name,
        pod_id,
        dataserver_endpoint,
        batch_data_ids,
        timeout=60,
    ):
        conn = self._connect(reader_leader_endpoint, timeout=30)

        req = data_server_pb2.ReportBatchDataMetaRequest()
        req.reader_name = reader_name
        req.pod_id = pod_id
        req.data_server_endpoint = dataserver_endpoint
        for batch_data_id in batch_data_ids:
            req.batch_data_ids.append(batch_data_id)

        with conn.lock:
            res = conn.stub.ReportBatchDataMeta(req)

        if res.status.type != "":
            exceptions.deserialize(res.status)

    @error_utils.handle_errors_until_timeout
    def reach_data_end(self, reader_leader_endpoint, reader_name, pod_id, timeout=60):
        conn = self.connect(reader_leader_endpoint, timeout=30)

        req = data_server_pb2.ReachDataEndRequest()
        req.reader_name = reader_name
        req.pod_id = pod_id

        with conn.lock:
            res = conn.stub.ReachDataEnd(req)

        if res.status.type != "":
            exceptions.deserialize(res.status)

    @error_utils.handle_errors_until_timeout
    def get_batch_data_meta(
        self, reader_leader_endpoint, reader_name, pod_id, timeout=60
    ):
        conn = self._connect(reader_leader_endpoint, timeout=30)

        req = data_server_pb2.GetBatchDataMetaRequest()
        req.reader_name = reader_name
        req.pod_id = pod_id

        with conn.lock:
            res = conn.stub.GetBatchDataMeta(req)

        if res.status.type != "":
            exceptions.deserialize(res.status)

        logger.debug(
            "pod client get_batch_data_meta:{}".format(
                pb_utils.batch_data_meta_response_to_string(res)
            )
        )
        return res.data

    @error_utils.handle_errors_until_timeout
    def get_batch_data(self, reader_leader_endpoint, req, timeout=60):
        """
        return BatchDataResponse
        """
        conn = self._connect(reader_leader_endpoint, timeout=30)

        with conn.lock:
            res = conn.stub.GetBatchData(req)
        if res.status.type != "":
            exceptions.deserialize(res.status)

        logger.debug(
            "pod client get batch_data meta:{}".format(
                pb_utils.batch_data_response_to_string(res)
            )
        )
        return res.data
