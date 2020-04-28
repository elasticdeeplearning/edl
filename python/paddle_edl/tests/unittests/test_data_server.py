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

import unittest
from paddle_edl.utils.data_server import DataServer
from paddle_edl.utils.dataset import TxtDataSet
import paddle_edl.utils.data_server_pb2_grpc as data_server_pb2_grpc
import paddle_edl.utils.data_server_pb2 as data_server_pb2
from paddle_edl.utils.utils import file_list_to_dataset
import time
import threading
import grpc


class TestDataServer(unittest.TestCase):
    def setUp(self):
        pass

    def test_data_server(self):
        endpoint = "0.0.0.0:6700"
        data_server = DataServer()
        data_server.start(
            endpoint=endpoint,
            data_set_reader=TxtDataSet(),
            file_list="./test_file_list.txt",
            master=None)
        print("start data server:", endpoint)
        time.sleep(3)

        channel = grpc.insecure_channel("127.0.0.1:6700")
        stub = data_server_pb2_grpc.DataServerStub(channel)

        request = data_server_pb2.DataRequest()
        for t in file_list_to_dataset('./test_file_list.txt'):
            meta = data_server_pb2.DataMeta()
            meta.idx_in_list = t.idx_in_list
            meta.file_path = t.file_path
            for i in range(3):
                meta.record_no.append(i)

            request.metas.append(meta)

        response = stub.GetData(request)
        print(response.files)
        """
        datas = []
        for data_file in response.files:
            datas.append(file_data_set)

        for data in datas:
            print("data:", data)
        """

        data_server.stop()

    def test_master(self):
        pass


if __name__ == '__main__':
    unittest.main()
