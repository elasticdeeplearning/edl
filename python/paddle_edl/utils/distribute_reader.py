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


class DistributeReader(object):
    def __init__(self, master=None):
        self.master = master
        if master == None:
            self._master = os.getenv("PADDLE_MASTER", "")
        assert self._master is not None, "master must be not empty"

    def get_meta(self, batch_size, step_num):
        pass

    def report(self, metas, success=True):
        pass

    def get_data(self, metas):
        pass
