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

import multiprocessing
from edl.uitls import reader as edl_reader
from edl.utils import batch_data_generator
from edl.utils import exceptions
from edl.utils import unique_name
from edl.utils.log_utils import logger


class Reader(object):
    def __init__(self,
                 state,
                 file_list,
                 file_splitter_cls,
                 batch_size,
                 #fields,
                 cache_capcity=100):
        self._file_list = file_list
        assert isinstance(self._file_list, list), "file_list must be a list"

        self._name = unique_name.generator("_dist_reader_")

        self._cls = file_splitter_cls
        self._batch_size = batch_size
        #self._fields = fields
        assert self._batch_size > 0, "batch size must > 0"
        self._cache_capcity = cache_capcity

        # reader meta
        self._reader_leader = edl_reader.load_from_ectd(
            self._etcd, self._trainer_env.pod_leader_id, timeout=60)

        self._generater_out_queue = multiprocessing.Queue(self._cache_capcity)
        self._accesser_out_queue = multiprocessing.Queue(self._cache_capcity)

        self._generater = None
        self._accesser = None

    def stop(self):
        if self._generater:
            self._generater.stop()
            self._generater = None

        if self._accesser:
            self._accesser.terminate()
            self._accesser.join()
            self._accesser = None

    def __exit__(self):
        self.stop()

    def _check_accesser(self):
        if self._accesser.is_alive():
            return True

        self._accesser.join()
        exitcode = self._accesser.exitcode
        if exitcode == 0:
            return False

        if len(self._error_queue) > 0:
            raise exceptions.EdlAccessDataError(self.error_queue[0])
        else:
            raise exceptions.EdlAccessDataError(
                "access process exit:{}".format(exitcode))

    def __iter__(self):
        self._generater = multiprocessing.Process(target=batch_data_generator.generate, args=args)
        self._generator.start()

        self._accesser = multiprocessing.Process(
            access_batch_data,
            args=(self._reader_leader, self._name, self._trainer_env,
                  self._generater_out_queue, self._accesser_out_queue,
                  self._cache_capcity))
        while True:
            if not self._check_accesser():
                break

            try:
                b = self._accesser_out_queue.pop(60)
            except multiprocessing.Queue.Empty as e:
                continue

            if b is None:
                logger.debug("{} reach data end".format(self._name))
                break
            yield {"meta": b[0], "data": b[1]}
