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
from exceptions import edl_exception


class Client(object):
    def __init__(self, endpoint):
        self._endpoint = endpoint

    def get_cluster(self, pod_id=None):
        pass

    def _get_conn(self):
        channel = grpc.insecure_channel(self._endpoint)
        stub = data_server_pb2_grpc.DataServerStub(channel)
        return channel, stub

    def add_dataset(self, dataset):
        channel = grpc.insecure_channel(self._endpoint)
        stub = master_pb2_grpc.MasterStub(channel)
        return stub.AddDataSet(dataset)

    def new_epoch(self):
        pass

    def barrier(self, job_id, pod_id, timeout=15):
        req = master_pb2.BarrierRequest()
        req.job_id = job_id
        req.pod_id = pod_id

        c, s = self._get_conn()
        begin = time.time()
        while True:
            res = s.Barrier(req)
            error = res.ret
            if error.type == "":
                return

            if error.type == "BarrierError":
                if time.time() - begin > timeout:
                    logger.debug("job_id:{} pod_id:{} barrier time out".format(
                        job_id, pod_id))
                    raise edl_exception(error.type, error.details)
                time.sleep(1)
                continue

            if error.type == 'PodDroppedError':
                log.info("job_id:{} pod_id:{} not exist in cluster, exit now!".
                         format(job_id, pod_id))
                sys.exit(0)

            raise edl_exception(error.type, error.detail)
