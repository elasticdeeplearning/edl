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

import multiprocessing
import threading

from edl.utils.log_utils import logger
from edl.utils import log_utils


class ProcessWrapper(object):
    def __init__(self, worker_func, args):
        self._stop = None
        self._worker = None

        self._lock = multiprocessing.Lock()
        self._stop = multiprocessing.Event()
        self._worker = multiprocessing.Process(target=worker_func, args=args)

    def start(self):
        log_file_name = "edl_{}_{}.log".format(self._class.__name__, os.getpid().log)
        log_utils.get_logger(log_level=20, log_file_name=log_file_name)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker:
            self._worker.join()
            self._worker = None

        logger.info("{} exit".format(self.__class__.__name__))

    def is_stopped(self):
        return self._worker is None or not self._worker.is_alive()

    def __exit__(self):
        self.stop()
