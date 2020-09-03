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


class Checkpoint(object):
    def __init__(self):
        self._model_path = None
        self._data_checkpoint = None
        pass

    def to_json(self):
        pass

    def from_json(self, s):
        pass

    def is_file_complete(self, idx, path):
        pass

    def filter_record_range(self, record_range):
        pass

    @staticmethod
    def save_to_etcd(self, etcd, model_path, data_checkpoint):
        pass

    @staticmethod
    @handle_timeout_errors
    def load_from_etcd(self, etcd):
        c = Checkpoint()
        pass
