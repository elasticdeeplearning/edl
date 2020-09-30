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
import os
import paddle_edl.utils.common_pb2 as common_pb2
import paddle_edl.utils.data_server_pb2 as data_server_pb2
import paddle_edl.utils.data_server_pb2_grpc as data_server_pb2_grpc
import time
import unittest
from edl.utils import file_utils
from edl.utils import log_utils
from edl.utils.data_server import DataServer
from edl.utils.dataset import TxtDataReader
from edl.utils.string_utils import bytes_to_string

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""


class TestDataServer(unittest.TestCase):
    def setUp(self):
        pass

    def _start_data_server(self, port):
        endpoint = "0.0.0.0:{}".format(port)
        data_server = DataServer()
        data_server.start(
            addr="0.0.0.0",
            port=port,
            data_set_reader=TxtDataReader,
            file_list="./test_file_list.txt",
            master=None,
        )
        print("start data server:", endpoint)
        time.sleep(3)
        return data_server, endpoint

    def _shut_down(self, data_server, stub):
        request = common_pb2.ShutDownRequest()
        stub.ShutDown(request)
        data_server.wait(2)

    def test_data_server(self):
        data_server, endpoint = self._start_data_server(9700)
        channel = grpc.insecure_channel("127.0.0.1:9700")
        stub = data_server_pb2_grpc.DataServerStub(channel)

        a = ["a0", "a1", "a2"]
        b = ["b0", "b1", "b2"]
        request = data_server_pb2.DataRequest()
        for t in file_utils.read_txt_lines("./test_file_list.txt"):
            request.idx_in_list = t[1]
            request.file_path = t[0]
            chunk = common_pb2.Chunk()
            chunk.meta.begin = 0
            chunk.meta.end = 2
            request.chunks.append(chunk)

            response = stub.GetData(request)
            f_d = response.file
            if f_d.file_path == "data_server/a.txt":
                assert f_d.idx_in_list == 0, "f_d.idx_in_list:{}".format(
                    f_d.idx_in_list
                )
                for c in f_d.data:
                    for r in c.records:
                        assert bytes_to_string(r.data) == a[r.record_no]
            elif f_d.file_path == "data_server/b.txt":
                assert f_d.idx_in_list == 1, "f_d.idx_in_list:{}".format(
                    f_d.idx_in_list
                )
                for c in f_d.data:
                    for r in c.records:
                        assert bytes_to_string(r.data) == b[r.record_no]

        self._shut_down(data_server, stub)

    def test_clear_cache(self):
        data_server, endpoint = self._start_data_server(9701)

        channel = grpc.insecure_channel("127.0.0.1:9701")
        stub = data_server_pb2_grpc.DataServerStub(channel)

        request = data_server_pb2.DataRequest()
        for t in file_utils.read_txt_lines("./test_file_list.txt"):
            request.idx_in_list = t[1]
            request.file_path = t[0]
            chunk = common_pb2.Chunk()
            chunk.meta.begin = 0
            chunk.meta.end = 2
            request.chunks.append(chunk)

            # clear
            response = stub.ClearDataCache(request)

            # get data
            response = stub.GetData(request)
            f_d = response.file
            for m in f_d.data:
                assert m.chunk.status == common_pb2.NOT_FOUND

        self._shut_down(data_server, stub)


if __name__ == "__main__":
    logger = log_utils.get_logger(10)
    unittest.main()
