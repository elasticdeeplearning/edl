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
import common_pb2
import common_pb2_grpc
from cluster import Cluster
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
                return res.cluster

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


def edl_barrier(master_dog, job_env, pod_env, timeout=15):
    c = Client(master_dog.get_master().endpoint)
    pb_cluster = e.barrier(job_env.job_id, pod_env.pod_id, timeout)
    cluster = Cluster()
    cluster.init_frim_pb(pb_cluster)

    pod = cluster.get_pod_by_id(pod_env.pod_id)
    return cluster, pod


def edl_initial_barrier(master_dog,
                        job_env,
                        pod_env,
                        init_pod_endpoints,
                        timeout=15):
    while True:
        cluster, pod = edl_barrier(master_dog, job_env, pod_env, timeout)
        pod_endpoints = cluster.get_pod_endpoints()
        if cluster.pods_num(
        ) != init_pod_num and cluster.job_stage == "INITIAL":
            logger.info(
                "wait init_pod_num:{} now:{} init_pod_ips:{} now:{}".format(
                    len(init_pod_endpoints),
                    len(pod_endpoints), init_pod_endpoints, pod_endpoints))

            time.sleep(15)
            continue

        return cluster, pod
