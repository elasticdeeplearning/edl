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
from . import pod_server_pb2 as pb
from . import pod_server_pb2_grpc as pb_grpc
from .client import Client
from .cluster import Cluster
from .exceptions import deserialize_exception, EdlBarrierError
from .utils import logger


class PodServerClient(Client):
    def __init__(self, endpoint):
        super(PodServerClient, self).__init__(endpoint)

    def connect(self):
        super(PodServerClient, self).connect()
        self._stub = pb_grpc.PodServerStub(self._channel)
        return self._channel, self._stub

    def shutdown(self):
        super(PodServerClient, self).shutdown()
        self._stub = None

    def barrier(self, job_id, pod_id, timeout=15):
        """
        try to barrier on master with other launchers until timeout
        """
        req = pb.BarrierRequest()
        req.job_id = job_id
        req.pod_id = pod_id

        c, s = self.connect()
        begin = time.time()
        while True:
            res = s.Barrier(req)
            if res.type == "":
                logger.info("barrier ok!")
                return

            deserialize_exception(res)
            if time.time() - begin > timeout:
                message = "job_id:{} pod_id:{} barrier time out".format(job_id,
                                                                        pod_id)
                deserialize_exception(res)
            time.sleep(1)
