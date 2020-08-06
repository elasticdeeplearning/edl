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

import paddle.fluid as fluid
from fluid.reader import DataLoader
from ..utils import data_server


class Record(object):
    def __init__(self):
        self.record_no = -1
        r.data = None


class DataLoader(ojbect):
    def __init__(reader_cls,
                 file_list,
                 feed_list=None,
                 places=None,
                 return_list=False,
                 batch_sampler=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 collate_fn=None,
                 num_workers=0,
                 use_buffer_reader=True,
                 use_shared_memory=True,
                 timeout=0,
                 worker_init_fn=None):
        """
        Reader_cls is the class name of dataset.See example in dataset.py
        file_list is the input data file list and it should be get by loader.For example, all data
        file is on local or on hdfs.
        This class:
        1. get data file list from the leader.
        2. parse records from reader_cls' object.
        3. if there's no data local, try to pull data from other dataserver or raise StopIteration.
        """
        self._name = unique_name.generate("_dataloader_")

        self._loader = DataLoader()
        self._start_data_server()

        # to control the cache size.
        self._data_queue = Queue(capcity)
        self._lock = Lock()
        self._file_list = file_list
        self._reader_cls = reader_cls

        assert type(reader_cls) == DataReader

        self._t_read_data = Thread(target=self._read_data)
        self._t_read_data.start()

    def _start_data_server(self):
        """
        start and register the data server
        """
        self._data_server = dataserver.DataServer()
        pass

    def __iter__(self):
        """
        get data from queue
        """
        pass

    def __next__(self):
        pass

    def _get_one_data_file(self):
        pass

    def _get_data(self):
        """
        get data from queue
        """
        pass

    def _get_file_key(self, idx, file_path):
        return "idx:{}_path:{}".format(idx, file_path)

        return key

    def _read_data(self):
        """
        read data into queue
        """
        while True:
            file_data_set = self._sub_data_set.get()
            if file_data_set is None:
                logger.info("terminated exit!")
                break

            rec_map = {}
            for one_range in file_data_set.filtered_records:
                for rec_no in range(one_range.begin, one_range.end + 1):
                    rec_map[rec.record_no] = one_range.status

            for rec_no, data in enumerate(
                    self._reader_cls(file_data_set.file_path)):
                if rec_no in rec_map and rec_map[
                        rec_no] == RecordStatus.PROCSSED:
                    continue

                logger.debug("read rec_no:{} data_len:{}".format(rec_no,
                                                                 len(data)))

                self._data_queue.put(1, block=True)
                key = self._get_file_key(file_data_set.idx_in_list,
                                         file_data_set.file_path)
                with self._lock:
                    if key not in self._data:
                        self._data[key] = {}
                    self._data[key][rec_no] = data
