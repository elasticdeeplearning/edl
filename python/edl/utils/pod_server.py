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
from concurrent import futures
import grpc
import sys
import os
import logging
from threading import Thread, Lock
from six.moves.queue import Queue
import traceback
from .exceptions import *
import signal
import threading
import copy
from .utils import logger
from . import common_pb2 as common_pb
from . import pod_server_pb2 as pod_server_pb
from . import pod_server_pb2_grpc as pod_server_pb_grpc
from .etcd_db import EtcdDB as db
from .exceptions import *


class PodServerServicer(pod_server_pb_grpc.PodServerServicer):
    def __init__(self, job_env, pod_id):
        # to control the cache size.
        self._lock = Lock()

        # stage => set(pod_id)
        self._barrier_in = {}

        self._pod_id = pod_id
        self._job_env = job_env

    def ScaleOut(self, request, context):
        status = common_pb.Status()
        pod = db.get_pod_leader()
        if pod.get_id != self._pod_id:
            status = serialize_exception(
                EdlLeaderError("this pod is not the leader"))
            return status

        return status

    def ScaleIn(self, request, context):
        status = common_pb.Status()
        pod = db.get_pod_leader()
        if pod.get_id != self._pod_id:
            status = serialize_exception(
                EdlLeaderError("this pod is not the leader"))
            return status

        return status

    def IsWorldChanged(self, request, context):
        """
        1. failed/disappeared
        2. new added?
        """
        pass

    def GetDiffPods(self, request, context):
        # return succeed, failed, added, inited
        pass

    def Barrier(self, request, context):
        """
        1. pods barrier on the cluster's current stage
        """
        res = pod_server_pb.BarrierResponse()

        cluster = db.get_cluster()
        if cluster == None:
            serialize_exception(
                res, EdlBarrierError("get current running cluster error"))
            return res

        ids = cluster.get_pods_ids_set()
        logger.debug(
            "get barrier request from job_id:{} pod_id:{} ids_set:{} leader:{}".
            format(request.job_id, request.pod_id, ids, cluster.pods[0].get_id(
            )))

        try:
            key = cluster.stage

            with self._lock:
                if key not in self._barrier_in:
                    self._barrier_in[key] = set()

                bd = self._barrier_in[leader.stage]
                bd.add(request.pod_id)

            if ids == bd:
                if len(rank_ids) < self._job_env.min_nodes:
                    message = "barrier's rank cluster now:{} is not enough of job's min nodes:{}".format(
                        ids, self._job_env.min_nodes())
                    serialize_exception(res, EdlBarrierError(message))
                    return res

                res.cluster_json = cluster.to_json()
                return res

            serialize_exception(
                res,
                EdlBarrierError("barrier's context:{}, now:{}".format(ids,
                                                                      bd)))
            return res
        except Exception as e:
            logger.debug("internal error:{} {}".format(e,
                                                       traceback.format_exc()))
            serialize_exception(res, EdlInternalError(str(e)))
            return res

    def ShutDown(self, request, context):
        pass


class PodServer(object):
    def __init__(self, job_env, pod_id):
        self._server = None
        self._port = None
        self._endpoint = None
        self._pod_id = pod_id
        self._job_env = job_env

    def start(self, job_env, pod, concurrency=20, max_workers=100):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        pod_server_pb_grpc.add_PodServerServicer_to_server(
            PodServerServicer(self._job_env, self._pod_id), server)

        self._port = server.add_insecure_port('{}:0'.format(pod.addr))
        assert self._port > 0, "data server start on endpoint:{} error, selected port is {}".format(
            pod.addr, self._port)
        self._endpoint = "{}:{}".format(pod.addr, self._port)

        server.start()
        self._server = server
        pod.port = self._port

        logger.info("start podserver at:{}".format(self._endpoint))

    def wait(self, timeout=None):
        if timeout is not None:
            self._server.stop(timeout)
            return
        self._server.wait_for_termination(timeout)
