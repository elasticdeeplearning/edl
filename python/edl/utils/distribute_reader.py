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
import threading
import time
import json
import uuid
import copy
import traceback
import six

from .utils import logger
from .pod import Pod
from ..discovery.etcd_client import EtcdClient

import etcd3
from .global_vars import *
from .cluster import Cluster
from .exceptions import EdlGenerateClusterError, EdlTableError
from .etcd_db import get_global_etcd

from .utils.edl_env import TrainerEnv
from .utils import handle_timeout_errors


class Connection(self):
    def __init__(self, endpoint, channel, stub):
        self._endpoint = endpoint
        self.channel = channel
        self.stub = stub


class DistributeReader(object):
    def __init__(self, file_list, file_splitter_cls, batch_size):
        self._file_list = file_list
        assert isinstance(self._file_list, list)

        self._cls = file_splitter_cls
        self._batch_size = batch_size
        assert self._batch_size > 0

        self._id = str(uuid.uuid1())

        # connections to data servers
        self._conns = {}

        self._trainer_env = TrainerEnv()

        self._data_server = DataServer(self._trainer_env, self._id)
        self._data_server.start()

        self._record_to_dist_reader_table()

        self._db = get_global_etcd(trainer_env.endpoints, trainer_env.job_id)
        self._wait_all_dist_readers()
        self._wait_dist_reader_leader()

    @handle_timeout_errors
    def _wait_dist_reader_leader(self, timeout=120):
        self._leader = self._db.get_dist_reader_leader()

    @handle_timeout_errors
    def _wait_all_dist_readers(self, timeout=120):
        self._readers = self._db.check_dist_readers()

    @handle_timeout_errors
    def _record_to_dist_reader_table(self, timeout=120):
        self._db.record_to_dist_reader_table(
            self._pod_id, self._data_server.endpoint, self._id)

    def stop(self):
        self._data_server.stop()

    def __exit__(self):
        self.stop()

    def __iter__(self):
        pass

    def _get_file_list(self):
        self

    def get_meta(self, batch_size, step_num):
        pass

    def report(self, metas, success=True):
        pass

    def _connect_data_server(self, endpoint):
        if meata.data_server not in self._data_servers:
            channel = grpc.insecure_channel(endpoint)
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
