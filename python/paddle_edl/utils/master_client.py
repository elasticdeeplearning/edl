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

import master_pb2
import master_pb2_grpc
import grpc


class Client(object):
    def __init__(self, endpoint):
        #self._endpoint = MasterFinder().get_master()
        self._endpoint = endpoint

    def get_cluster(self, pod_id=None):
        pass

    def add_dataset(self, dataset):
        channel = grpc.insecure_channel(self._endpoint)
        stub = master_pb2_grpc.MasterStub(channel)

        # get data
        response = stub.AddDataSet(dataset)
        print("response:", response)
        return response

    def new_epoch(self):
        pass
