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

from __future__ import print_function

import concurrent
import grpc
import threading
import traceback
from edl.utils import cluster as edl_cluster
from edl.utils import common_pb2
from edl.utils import etcd_db
from edl.utils import exceptions
from edl.utils import leader_pod
from edl.utils import pod_server_pb2
from edl.utils import pod_server_pb2_grpc
from edl.utils import status as edl_status
from edl.utils.log_utils import logger


class PodServerServicer(pod_server_pb2_grpc.PodServerServicer):
    def __init__(self, job_env, pod_id):
        # to control the cache size.
        self._lock = threading.Lock()

        # stage => set(pod_id)
        self._barrier_in = {}

        self._pod_id = pod_id
        self._job_env = job_env

        self._etcd = etcd_db.get_global_etcd(
            self._job_env.etcd_endpoints, self._job_env.job_id
        )

    def ScaleOut(self, request, context):
        status = common_pb2.Status()
        pod = leader_pod.load_from_etcd(self._etcd)
        if pod.get_id != self._pod_id:
            status = exceptions.serialize(
                exceptions.EdlLeaderError("this pod is not the leader")
            )
            return status

        return status

    def ScaleIn(self, request, context):
        status = common_pb2.Status()
        pod = leader_pod.load_from_etcd(self._etcd)
        if pod.get_id != self._pod_id:
            status = exceptions.serialize(
                exceptions.EdlLeaderError("this pod is not the leader")
            )
            return status

        return status

    def Barrier(self, request, context):
        res = pod_server_pb2.BarrierResponse()

        try:
            cluster = edl_cluster.load_from_etcd(self._etcd, timeout=60)
            if cluster is None:
                exceptions.serialize(
                    res, exceptions.EdlBarrierError("get current running cluster error")
                )
                return res

            if cluster.status == edl_status.Status.FAILED:
                exceptions.serialize(
                    res, exceptions.EdlBarrierError("cluster's status is status.Failed")
                )
                return res

            ids = cluster.get_pods_ids_set()
            logger.debug(
                "get barrier request from job_id:{} pod_id:{} cluster table ids is {}".format(
                    request.job_id, request.pod_id, ids
                )
            )

            key = cluster.stage

            with self._lock:
                if key not in self._barrier_in:
                    self._barrier_in[key] = set()

                bd = self._barrier_in[key]
                bd.add(request.pod_id)

            if ids == bd:
                res.cluster_json = cluster.to_json()
                return res

            exceptions.serialize(
                res,
                exceptions.EdlBarrierError(
                    "barrier's context:{}, now:{}".format(ids, bd)
                ),
            )
            return res
        except Exception as e:
            logger.debug("internal error:{} {}".format(e, traceback.format_exc()))
            exceptions.serialize(res, exceptions.EdlInternalError(str(e)))
            return res

    def ShutDown(self, request, context):
        pass


class PodServer(object):
    def __init__(self, job_env, pod):
        self._server = None
        self._port = None
        self._endpoint = None
        self._pod = pod
        self._job_env = job_env

    def start(self, concurrency=20, max_workers=100):
        server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ],
            maximum_concurrent_rpcs=concurrency,
        )
        pod_server_pb2_grpc.add_PodServerServicer_to_server(
            PodServerServicer(self._job_env, self._pod.pod_id), server
        )

        self._port = server.add_insecure_port("{}:0".format(self._pod.addr))
        assert (
            self._port > 0
        ), "data server start on endpoint:{} error, \
            selected port is {}".format(
            self._pod.addr, self._port
        )

        self._endpoint = "{}:{}".format(self._pod.addr, self._port)

        server.start()
        self._server = server
        self._pod.port = self._port

        logger.info("start podserver at:{}".format(self._endpoint))

    def wait(self, timeout=None):
        if timeout is not None:
            self._server.stop(timeout)
            return
        self._server.wait_for_termination(timeout)
