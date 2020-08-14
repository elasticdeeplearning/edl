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
from fluid.reader import FileSplitter
from ..utils import data_server


class Record(object):
    def __init__(self, idx, data):
        # idx in a file
        self._idx = idx
        self._data = data


class FileMeta(object):
    def __init__(self, idx, path):
        self._idx = idx
        self._path = path


class BatchData(object):
    def __init__(self, data_reader_id, name):
        self._data_reader_id = data_reader_id
        self._name = name
        # FileIdx->Records
        self._batch = {}
        self._size = None


class DataCheckpoint(object):
    def __init__(self):
        # file_dix=>record_idx
        self._records = {}
        self._file_idxs = {}
        self._restored_from = None

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass

    def is_processed(self, path, file_idx, record_idx):
        pass


class DataReader(ojbect):
    def __init__(file_splitter_cls, file_list, batch_size, leader,
                 checkpoint_path):
        """
        file_splitter_cls is the class name of dataset.See example in dataset.py
        file_list is the input data file list and it should be get by loader.For example, all data
        file is on local or on hdfs.
        This class:
        1. get data file list from the leader.
        2. parse records from reader_cls' object.
        3. if there's no data local, try to pull data from other dataserver or raise StopIteration.
        """
        self._name = unique_name.generate("_datareader_")

        #BatchData
        self._data_queue = Queue(capcity)
        self._lock = Lock()
        self._file_list = file_list
        self._splitter_cls = file_splitter_cls
        self._leader = leader

        self._data_checkpoint = DataCheckpoint()
        self._data_checkpoint.load_checkpoint(checkpoint_path)

        assert type(file_splitter_cls) == FileSplitter

        self._t_read_data = Thread(target=self._read_data)
        self._t_read_data.start()

        self._start_data_server()

    def _start_data_server(self):
        """
        start and register the data server
        """
        self._data_server = data_server.DataServer()
        pass

    def _shut_down(self):
        pass

    def __iter__(self):
        """
        get data from queue
        """
        pass

    def __next__(self):
        if self._reach_end:
            raise StopIteration

        idx = {}
        data = []
        while True:
            idx, data = self._data_queue.Pop()
            if idx is None:
                self._reach_end = True
                break

        if len(idx) > 0:
            yield idx, data
        else:
            raise StopIteration

    def _get_one_data_file(self):
        pass

    def _get_file_key(self, idx, file_path):
        return "idx:{}_path:{}".format(idx, file_path)

        return key

    def _get_file_idx():
        """
        get file idx
        """
        while True:
            file_data_set = self._get_sub_dataset()
            if file_data_set is None:
                logger.info("local data process completed.")
                break
        pass

    def _set_batch_data(self, meta):
        """
        get batch data idx
        """
        meta = self._data_client.get_batch_data_meta(self)
        # reach end
        if meta is None:
            self._data_queue.push(None)
        self._data_queue.push(b)

        if meta.is_local():
            b = self._cache[meta.name]
        else:
            b = self._data_client.get_batch_data(meta)
        self._data_queue.push(b)

    def _process_one_file(self, idx, path):
        for rec_no, data in enumerate(self._splitter_cls(path)):
            if self._data_checkpoint.is_processed(path, idx, rec_no):
                logger.debug(
                    "idx:{} file:{} rec_no:{} data_len:{} already processed".
                    format(idx, path, rec_no, len(data)))
                continue

            logger.debug("read idx:{} file:{} rec_no:{} data_len:{}".format(
                idx, path, rec_no, len(data)))

            yield Record(rec_no, data)

    def _new_batch_data(self):
        pass

    def _process_file_list(self, metas):
        rec_map = {}
        b = self._new_batch_data()
        size = 0
        for m in metas:
            for rec in self._process_one_file(m._idx, m._path):
                if m not in b._batch:
                    b._batch[m] = []
                b._batch[m].append(rec)
                size += 1
        b._batch._size = size
        yield b

    def _report_batch_data(self, batch_data):
        pass

    def _read_data(self):
        """
        read data into queue
        """
        while True:
            file_list = self._data_client._get_file_list()
            if file_data_set is not None:
                for batch_data in self._process_file_list(file_list):
                    self._report_batch_data(batch_data)
                    meta = self._data_client.get_batch_data_meta(self)
                    self._set_batch_data(meta)
                continue

            break

        logger.info("local data process completed.")
        self._set_batch_data(meta)
