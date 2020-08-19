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
from .exceptions import *
import signal
import threading
import copy
from .utils import logger
from . import pod_server_pb2_grpc as pb2_grpc
from . import pod_server_pb2 as pb
from . import common_pb2 as common_pb
from .watcher import get_current_pod_ids_from_resource, get_pod_leader
from .exceptions import *


class PodServerServicer(pb2_grpc.PodServerServicer):
    def __init__(self, rank_register):
        # to control the cache size.
        self._lock = Lock()
        # stage => set(pod_id)
        self._barrier_in = {}
        self._rank_register = rank_register

    def ScaleOut(self, request, context):
        status = common_pb.Status()
        if self._rank_register.rank != 0:
            status = serialize_exception(
                EdlLeaderError("this pod is not the leader"))
            return status

        return status

    def ScaleIn(self, request, context):
        status = common_pb.Status()
        if self._rank_register.rank != 0:
            status = serialize_exception(
                EdlLeaderError("this pod is not the leader"))
            return status

        return status

    def Barrier(self, request, context):
        ids = get_current_pod_ids_from_resource()
        leader = get_pod_leader()
        logger.info(
            "get barrier request from job_id:{} pod_id:{} ids_set:{} leader:{}".
            format(request.job_id, request.pod_id, ids, leader.get_id()))

        status = common_pb.Status()
        with self._lock:
            try:
                key = leader.stage
                if key not in self._barrier_in:
                    self._barrier_in[key] = set()

                bd = self._barrier_in[leader.stage]
                bd.add(request.pod_id)

                if ids == bd:
                    return status

                status = serialize_exception(
                    EdlBarrierError("barrier's context:{}, now:{}".format(ids,
                                                                          bd)))
                return status
            except Exception as e:
                status = serialize_exception(EdlInternalError(str(e)))
                return status

    def ShutDown(self, request, context):
        pass


class PodServer(object):
    def __init__(self, rank_register):
        self._server = None
        self._port = None
        self._endpoint = None
        self._rank_register = rank_register

    def start(self, job_env, pod, concurrency=20, max_workers=100):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        pb2_grpc.add_PodServerServicer_to_server(
            PodServerServicer(rank_register), server)

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
