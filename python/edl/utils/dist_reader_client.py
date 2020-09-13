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


class DistReaderClient(Client):
    def __init__(self):
        super(DistReaderClient, self).__init__(endpoint)

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
