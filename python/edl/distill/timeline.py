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

import os
import sys
import time


class _NopTimeLine(object):
    def record(self, name):
        pass

    def reset(self):
        pass


class _RealTimeLine(object):
    def __init__(self):
        self.pid = os.getpid()
        self.time = time.time()

    def record(self, name):
        new_time = time.time()
        sys.stderr.write('pid={} op={} time={}ms\n'.format(self.pid, name, (
            new_time - self.time) * 1000))
        self.time = new_time

    def reset(self):
        self.time = time.time()


_is_profile = int(os.environ.get('DISTILL_READER_PROFILE', 0))
_TimeLine = _RealTimeLine if _is_profile else _NopTimeLine
