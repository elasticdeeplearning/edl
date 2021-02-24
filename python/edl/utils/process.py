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


class ProcessWrapper(object):
    def __init__(self):
        self._stop = None
        self._lock = None
        self._worker = None

        self._stop = multiprocessing.Event()
        self._lock = threading.Lock()
        self._worker = multiprocessing.Process(target=self._worker_func)

    def _worker_func(self):
        raise NotImplementedError

    def start(self):
        self._worker.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._worker:
                self._worker.join()
                self._worker = None

        logger.info("{} exit".format(self.__class__.__name__))

    def is_stopped(self):
        with self._lock:
            return self._worker is None

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
