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
from edl.utils import reader as edl_reader
from edl.utils import batch_data_generator
from edl.utils import batch_data_accesser
from edl.utils import exceptions
from edl.utils import unique_name
from edl.utils.log_utils import logger


class Reader(object):
    def __init__(
        self, state, file_list, file_splitter_cls, batch_size, cache_capcity=100,
    ):
        self._file_list = file_list
        assert isinstance(self._file_list, list), "file_list must be a list of string"
        self._state = state

        self._name = unique_name.generator("_dist_reader_")

        self._cls = file_splitter_cls
        self._batch_size = batch_size

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

        self._logger_no = 0

    def _terminate_process(self, proc):
        if proc is None:
            return

        proc.terminate()
        proc.join()
        proc = None

    def stop(self):
        self._terminate_process(self._generator)
        self._terminate_process(self._accesser)

    def __exit__(self):
        self.stop()

    def _check_proc(self, proc, error_queue):
        if self.proc.is_alive():
            return True

        self.proc.join()
        exitcode = self.proc.exitcode
        if exitcode == 0:
            return False

        if len(error_queue) > 0:
            raise exceptions.EdlDataProcessError(error_queue[0])
            return

        raise exceptions.EdlDataProcessError("process exit:{}".format(exitcode))

    def _start_generator(self):
        args = batch_data_generator.Args()
        args.state = self._state
        args.reader_leader_endpoint = self._reader_leader.endpoint
        args.reader_name = self._reader_leader.name
        args.pod_id = self._pod_id
        args.all_file_list = self._file_list
        args.splitter_cls = self._splitter_cls
        args.out_queue = self._generater_out_queue
        args.error_queue = self._generater_error_queue
        args.loger_name = "{}_generator_{}.log".format(self._name, self._logger_no)
        logger.debug("start generator args {}".format(args))

        self._generator = multiprocessing.Process(
            target=batch_data_generator.generate, args=args
        )
        self._generator.start()

    def _start_accesser(self):
        args = batch_data_accesser.Args()
        args.reader_leader_endpoint = self._reader_leader.endpoint
        args.reader_name = self._reader_leader.name
        args.input_queue = self._generater_out_queue
        args.trainer_env = self._trainer_env
        args.out_queue = self._accesser_out_queue
        args.queue_size = self._cache_capcity
        args.loger_name = "{}_accesser_{}.log".format(self._name, self._logger_no)
        logger.debug("start accesser args {}".format(args))

        self._accesser = multiprocessing.Process(
            batch_data_accesser.generate, args=(args)
        )
        self._accesser.start()

    def __iter__(self):
        self._start_generator()
        self._start_accesser()
        self._logger_no += 1

        while True:
            if not self._check_proc(self._accesser, self._accesser_error_queue):
                break

            if not self._check_proc(self._generator, self._generater_error_queue):
                break

            try:
                b = self._accesser_out_queue.pop(10)
            except multiprocessing.Queue.Empty:
                continue

            if b is None:
                logger.info("distributed reader {} reach data end".format(self._name))
                break
            yield {"meta": b[0], "data": b[1]}
