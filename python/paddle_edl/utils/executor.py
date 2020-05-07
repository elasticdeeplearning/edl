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

import paddle
import paddle.fluid.executor.Executor as PaddleExecutor
import Queue
from meta_reader import MetaReader


class Executor(PaddleExecutor):
    def __init__(self, place, master, step_num=10):
        func = PaddleExecutor.__init__
        assert func.__code__.co_varnames == ('self', 'place'), \
            "The executor of paddle is changed and executor of edl hasn't been changed."

        super(Executor, self).__init__(place)

        self._step_id = -1l
        self._step_num = step_num

        self._meta_queue = Queue()
        self._meta = MetaReader(master=master)

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
            use_program_cache=False,
            return_merged=True,
            use_prune=False):
        assert feed != None, "In EDL feed must not be empty"

        real_feed = feed[1:]

        super(Executor, self).run(program=program,
                                  feed=feed,
                                  fetch_list=fetch_list,
                                  feed_var_name=feed_var_name,
                                  fetch_var_name=fetch_var_name,
                                  scope=scope,
                                  return_numpy=return_numpy,
                                  use_program_cache=use_program_cache,
                                  return_merged=return_merged,
                                  use_prune=user_prune)

        self._step_id += 1
        self._meta_queue.push_back((self._step_id, meta))
        if self._step_id % self._step_num == 0 and self._step_id > 0:
            self._report_meta(self)

    def _report_meta(self):
        metas = []
        while not self._meta_queue.empty():
            metas.append(self._meta_queue.get()[1])

        self._meta.report(metas)
