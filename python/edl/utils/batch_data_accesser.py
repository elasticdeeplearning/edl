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

import threading

from edl.utils import reader as edl_reader
from edl.utils import data_server
from edl.utils import data_server_client
from edl.utils import etcd_db

logger = None


class Args(object):
    def __init__(self):
        self.reader_leader_endpoint = None
        self.reader_name = None
        self.trainer_env = None
        self.input_queue = None
        self.out_queue = None
        self.queue_size = None
        self.error_queue = None


class Accesser(object):
    """
    1. get data from batch_data_generator
    2. get batch_data_meta from data_server_leader
    3. get batch_data by batch_data_meta
    """

    def __init__(self, args):
        self._reader_leader_endpoint = args.reader_leader_endpoint

        self._reader_name = args.reader_name
        self._trainer_env = args.trainer_env
        # self._etcd = None

        # BatchData
        self._input_queue = args.input_queue
        self._out_queue = args.out_queue
        # batch_data_id => BatchData
        self._cache = {}

        # pb.BatchDataRequest queue
        self._req_queue = threading.Queue(args.queue_size)

        self._data_server = None

        self._stop = threading.Event()
        # self._t_reporter = threading.Thread(target=self._report)
        self._t_generater = threading.Thread(target=self._generate)
        self._t_accesser = threading.Thread(target=self._access)

        self._client = data_server_client.Client()

    def start(self):
        try:
            self._start()
        finally:
            self._stop.set()
            self.__exit__()

    def __exit__(self):
        # if self._t_reporter is not None:
        #    self._t_reporter.join()

        if self._t_generater is not None:
            self._t_generater.join()

        if self._t_accesser is not None:
            self._t_accesser.join()

        # self._t_reporter = None
        self._t_accesser = None
        self._t_generater = None

    def _start(self):
        self._data_server = data_server.Server(self)
        self._data_server.start()

        etcd = etcd_db.get_global_etcd(
            self._trainer_env.etcd_endpoint, job_id=self._trainer_env.job_id
        )

        edl_reader.save_to_etcd(
            etcd,
            reader_name=self._reader_name,
            pod_id=self._trainer_env.pod_id,
            data_server_endpoint=self._data_server.endpoint,
            timeout=30,
        )

        self._client.connect(self._reader_leader_endpoint)
        # self._t_reporter.start()
        self._t_generater.start()
        self._t_accesser.start()

    def _access(self):
        while not self._stop.set():
            res = self._client.get_batch_data_meta(
                reader_leader_endpoint=self._reader_leader_endpoint,
                reader_name=self._name,
                pod_id=self._trainer_env.pod_id,
            )

            self._req_queue.put(res)

            # data end
            if res is None:
                break

    def _get_batch_data(self, req):
        """
        Read BatchData from local or remote by BatchDataRequest
        """
        if self._trainer_env.pod_id != req.producer_pod_id:
            return (req, self._client.get_batch_data(req))

        return (req, self.get_local_batch_data(req))

    def get_local_batch_data(self, req):
        ret = []
        for batch_data_id in req.data.batch_data_ids:
            with self._lock:
                ret.append(self._cache.pop(batch_data_id))

        return ret

    def _generate(self):
        while not self._stop.set():
            req = self._req_queue.pop()
            if req is None:
                break

            ret = self._get_batch_data(req)
            for b in ret:
                self._out_queue.put(b)

        self._out_queue.put(None)


def generate(args):
    from edl.utils import log_utils

    global logger
    logger = log_utils.get_logger(log_level=20, log_file_name=args.loger_file_name)
    logger.info("args:{}".format(args))

    try:
        accesser = Accesser(args)
        accesser.start()
    except Exception:
        import traceback

        args.error_queue.put(traceback.format_exc())
