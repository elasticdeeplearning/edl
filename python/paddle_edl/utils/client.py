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

from . import data_server_pb2 as pb
from . import data_sever_pb2_grpc as pb_grpc
from .cluster import Cluster
import grpc
from .exceptions import raise_execption


class Client(object):
    def __init__(self, endpoint):
        self._endpoint = endpoint

    def connect(self):
        self._channel = grpc.insecure_channel(self._endpoint)
        self._stub = data_server_pb2_grpc.DataServerStub(channel)

    def stop(self):
        self._channel = None
        self._stub = None


class DataClient(Client):
    def __init__(self):
        super(DataClient, self).__init__(endpoint)

    def _request(self, server, req):
        res = stub.GetFileList(req)
        if res.status.type != "":
            raise_exeception(res.status.type, res.status.detail)

        return res

    def get_file_list(self, server, self_id):
        req = pb.FileListRequest()
        req.data_reader_id = self_id

        res = request(req)
        return res.metas

    def get_batch_data_meta(self, server, self_id, batch_id):
        req = pb.BatchDataMetaRequest()
        req.data_reader_id = self_id
        req.batch_id = batch_id

        res = request(server, req)
        return res.meta

    def get_batch_data(self, server, batch_id):
        req = pb.BatchDataRequest()
        req.batch_id = batch_id

        res = request(server, req)
        return res.batch


'''
    def _get_conn(self):
        channel = grpc.insecure_channel(self._endpoint)
        stub = data_server_pb2_grpc.DataServerStub(channel)
        return channel, stub

    def add_dataset(self, dataset):
        channel = grpc.insecure_channel(self._endpoint)
        stub = master_pb2_grpc.MasterStub(channel)
        return stub.AddDataSet(dataset)

    def barrier(self, job_id, pod_id, timeout=15):
        """
        try to barrier on master with other launchers until timeout
        """
        req = master_pb2.BarrierRequest()
        req.job_id = job_id
        req.pod_id = pod_id

        c, s = self._get_conn()
        begin = time.time()
        while True:
            res = s.Barrier(req)
            error = res.ret
            if error.type == "":
                return True

            if error.type == "BarrierError":
                if time.time() - begin > timeout:
                    message = "job_id:{} pod_id:{} barrier time out".format(
                        job_id, pod_id)
                    raise EdlBarrierError(message)
                time.sleep(1)
                continue

            raise EdlExeception(error.detail)
'''
