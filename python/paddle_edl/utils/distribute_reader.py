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


class Connection(self):
    def __init__(self, endpoint, channel, stub):
        self._endpoint = endpoint
        self.channel = channel
        self.stub = stub


class DistributeReader(object):
    def __init__(self):
        self._master = MasterFinder().get_master()
        self._data_servers = {}

    def get_meta(self, batch_size, step_num):
        pass

    def report(self, metas, success=True):
        pass

    def _connect_master(self, endpoint):
        pass

    def _connect_data_server(self, endpoint):
        if meata.data_server not in self._data_servers:
            channel = grpc.insecure_channel("127.0.0.1:6700")
            stub = data_server_pb2_grpc.DataServerStub(channel)
            conn = Connection(endpoint, channel, stub)
            self._data_servers[endpoint] = conn
            return conn

        return self._data_servers[endpoint]

    def get_data(self, meta):
        conn = self._connect_data_server(endpoint)
        response = conn.stub.GetData(request)

        if len(response.errors.errors) > 1:
            return

        data = []
        for f in response.files.files:
            for rec in f.records:
                data.append(rec.data)

        return data
