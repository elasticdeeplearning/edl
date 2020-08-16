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

from six.moves.queue import Queue
import multiprocessing


class FileSplitter():
    """
    This the interface user should inherit. It will let's the framework knows the data file it's
    processing.
    TxtDataReader is an example.
    """

    def __init__(self, data_file):
        self._data_file = data_file

    def __iter__(self):
        """
        yield idx, record data
        """
        raise NotImplementedError()


class TxtFileSplitter(FileSplitter):
    def __init__(self, data_file):
        super(self, TxtFileSplitter).__init__(data_file)

    def __iter__(self):
        idx = 0
        with open(self._data_file, "rb") as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    continue
                idx += 1
                yield idx, line
