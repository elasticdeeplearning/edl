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
from six.moves.queue import Queue
from meta_reader import MetaReader


class Executor(PaddleExecutor):
    """
    Executor wrapper of Paddle' Executor.
    1. Trainers' leader will tell others to save checkpoint.
    2. All trainers can load checkpoint from given backend.
    """

    def __init__(self, place):
        func = PaddleExecutor.__init__
        assert func.__code__.co_varnames == ('self', 'place'), \
            "The executor of paddle is changed and executor of edl hasn't been changed."

        super(Executor, self).__init__(place)

        self._processed_data = {}
        self._name = unique_name.generate("_executor_")

        self._step_inter = None

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
        """
        Feed has not only the data but also the dataloader object and the data's index.
        So executor can save the data checkpoint.Before run, the part should shoud split from feed.
        """
        assert feed != None, "In EDL feed must not be empty"

        # it's a barrier function on multiple trainers.
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

        self._add_processed_data()
        self._save_checkpoint()

    def _add_processed_data(self):
        pass

    def _prepare_save_checkpoint(self):
        """
        Ask all the trainers to prepare save_checkpoint, and make sure it's succeed or exit.
        """
        pass

    def _save_data_checkpoint(self):
        pass

    def _save_model_checkpoint(self):
        pass

    def _end_save_checkpoint(self):
        """
        Wait all the trainers to save_checkpoint, and make sure it's succeed or exit.
        """
        pass

    def _save_checkpoint_transaction(self):
        """
        trainer 0 save model checkpoint and all trainers save data checkpoint.
        It's a tranaction.
        """
        pass
