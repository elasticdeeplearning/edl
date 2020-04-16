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
import paddle.fluid as fluid


class PaddleExecutor(object):
    def __init__(self, place):
        self._exe = fluid.Executor(place)

    def init(program):
        return self._exe.run(program)

    def train(program, data, fetch_list):
        fetched = train_exe.run(program, feed=data, fetch_list=fetch_list)
        return fetched
        """
        ret = []
        for item in fetched:
            ret.append(numpy(item))
        """

    def inference(program, data, fetch_list):
        pass
