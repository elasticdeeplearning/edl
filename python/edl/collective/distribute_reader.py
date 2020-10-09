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
from edl.utils import batch_data_accesser
from edl.utils import exceptions
from edl.utils import unique_name
from edl.utils.log_utils import logger


class Reader(object):
    def __init__(
        self,
        state,
        file_list,
        file_splitter_cls,
        batch_size,
        # fields,
        cache_capcity=100,
    ):
        self._file_list = file_list
        assert isinstance(self._file_list, list), "file_list must be a list"

        self._name = unique_name.generator("_dist_reader_")

        self._cls = file_splitter_cls
        self._batch_size = batch_size
        # self._fields = fields
        assert self._batch_size > 0, "batch size must > 0"
        self._cache_capcity = cache_capcity

        # reader meta
        self._reader_leader = edl_reader.load_from_ectd(
            self._etcd, self._trainer_env.pod_leader_id, timeout=60
        )

        self._generater = None
        self._generater_out_queue = multiprocessing.Queue(self._cache_capcity)
        self._generater_error_queue = multiprocessing.Queue()

        self._accesser = None
        self._accesser_out_queue = multiprocessing.Queue(self._cache_capcity)
        self._accesser_error_queue = multiprocessing.Queue()

    def stop(self):
        if self._generater:
            self._accesser.terminate()
            self._accesser.join()
            self._generater = None

        if self._accesser:
            self._accesser.terminate()
            self._accesser.join()
            self._accesser = None

    def __exit__(self):
        self.stop()

    @staticmethod
    def _check_processor(process, error_queue):
        if process.is_alive():
            return True

        process.join()
        exitcode = process.exitcode
        if exitcode == 0:
            return False

        if len(error_queue) > 0:
            raise exceptions.EdlAccessDataError(error_queue[0])
        else:
            raise exceptions.EdlAccessDataError("process exit:{}".format(exitcode))

    def __iter__(self):
        args = batch_data_generator.Args()
        self._generater = multiprocessing.Process(
            target=batch_data_generator.generate, args=args
        )
        self._generator.start()

        args = batch_data_accesser.Args()
        self._accesser = multiprocessing.Process(
            batch_data_accesser.access_batch_data, args=args,
        )

        while True:
            self._check_processor(self._generater, self._generater_error_queue)

            if not self._check_process(self._accesser, self._accesser_error_queue):
                break

            try:
                b = self._accesser_out_queue.pop(60)
            except multiprocessing.Queue.Empty:
                continue

            if b is None:
                logger.info("{} reach data end".format(self._name))
                break
            yield {"meta": b[0], "data": b[1]}
