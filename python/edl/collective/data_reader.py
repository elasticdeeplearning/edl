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
import uuid


class Record(object):
    def __init__(self, idx, data):
        # idx in a file
        self._idx = idx
        self._data = data  # data is ()


class FileMeta(object):
    def __init__(self, idx, path):
        self._idx = idx
        self._path = path


class BatchData(object):
    def __init__(self, data_reader_id, b_id):
        self._data_reader_id = data_reader_id
        self._id = b_id
        # FileMeta->Records
        self._batch = {}
        self._size = None

    def split_meta_and_data(object):
        b = BatchData(self.data_reader_id, self._id)
        b._size = self._size

        a = []
        for fmeta, recs in self._batch:
            rs = []
            for rec in recs:
                r = Record(rec._idx, None)
                a.append(rec.data)
                rs.append(r)
            b._batch[fmeta] = rs

        return b, a


class DataCheckpoint(object):
    def __init__(self):
        # file_idx=>set(record_idx)
        self._restored_records = {}
        #self._file_idxs = {}
        self._restored_from = None

    def save_checkpoint(self, path, batch_datas):
        pass

    def load_checkpoint(self, path):
        pass

    def is_processed(self, file_idx, path, record_idx):
        if file_idx not in self._restored_records:
            return False

        rec_idxs = self._restored_records[file_idx]
        if record_idx not in rec_idxs:
            return False

        return True


class FileCache(object):
    def __init__(self, capcity=100):
        """
        capcity: GB
        """
        pass

    def download(self, src_path, dst_path):
        pass

    def _clean_up(self):
        pass


class DistributedDataReader(ojbect):
    def __init__(file_list,
                 file_splitter_cls,
                 splitted_data_field,
                 batch_size,
                 capcity=100):
        """
        file_splitter_cls is the class name of dataset.See example in dataset.py
        file_list is the input data file list and it should be get by loader.For example, all data
        splitted_data_field: the  file_splitter_cls's result field name by order
        file is on local or on hdfs.
        This class:
        1. get data file list from the leader.
        2. parse records from reader_cls' object.
        3. if there's no data local, try to pull data from other dataserver or raise StopIteration.

        capcity: cached batch num

        __next__: return meta, (splitted_data_field data)
        """
        """
        self._job_id = os["PADDLE_JOB_ID"]
        self._etcd_endpoints = os["PADDLE_ETCD_ENDPOINTS"]
        self._rank = os["PADDLE_EDL_RANK"]
        """

        self._id = uuid.uuid1()
        self._name = unique_name.generate("_datareader_")
        #self._rank = trainer_rank

        #BatchData
        self._data_queue = Queue(capcity)

        self._lock = Lock()
        self._file_list = file_list
        self._splitter_cls = file_splitter_cls
        self._leader = leader

        self._data_checkpoint = DataCheckpoint()
        self._data_checkpoint.load_checkpoint(checkpoint_path)
        self._reach_end = False
        self._cache = {}
        self._file_cache = FileCache()
        self._b_id = 0

        assert type(file_splitter_cls) == FileSplitter

        self._register = DataReaderRegister(
            etcd_endoints=self._etcd_endpoints,
            job_id=self._job_id,
            rank=self._rank,
            reader=self)

        self._t_read_data = Thread(target=self._read_data)
        self._t_read_data.start()

        self._start_data_server()

    def get_port(self):
        return self._data_server.port()

    def _start_data_server(self):
        """
        start and register the data server
        """
        self._data_server = data_server.DataServer(self._file_list, world_rank)

    def _shut_down(self):
        self._data_server.wait()
        pass

    def __iter__(self):
        """
        get data from queue
        """
        self._local_file_list = self._data_client._get_file_list()
        self._reach_end = False
        if self._t_read_data is None:
            self._t_read_data = Thread(target=self._read_data)
            self._t_read_data.start()

    def __next__(self):
        while True:
            b = self._data_queue.pop()
            if b is None:
                break
            yield b.split_meta_and_data()  # meta, data

        self._t_read_data.join()
        self._t_read_data = None
        self._reach_end = True
        raise StopIteration

    def _set_batch_data(self, meta):
        """
        get batch data meta
        """
        # reach end
        if meta is None:
            self._data_queue.put(None)
            return False

        if meta.is_local():
            b = self._cache.pop(meta._id)
        else:
            b = self._data_client.get_batch_data(meta)
        self._data_queue.put(b)
        if b is None:
            return False
        return True

    def _process_one_file(self, idx, file_path):
        path = self._file_cache.download(file_path)
        for r in enumerate(self._splitter_cls(path)):
            rec_no = r[0]
            data = r[1:]
            if self._data_checkpoint.is_processed(idx, path, rec_no):
                logger.debug(
                    "idx:{} file:{} rec_no:{} data_len:{} already processed".
                    format(idx, path, rec_no, len(data)))
                continue

            logger.debug("read idx:{} file:{} rec_no:{} data_len:{}".format(
                idx, path, rec_no, len(data)))

            yield Record(rec_no, (data))

    def _new_batch_data(self):
        self._b_id += 1
        b = BatchData(self._id, self._b_id)
        return b

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
                if size >= self._batch_size:
                    b._batch._size = size
                    yield b
                    size = 0
                else:
                    continue

        if size > 0:
            yield b

    def _read_data(self):
        """
        read data into queue
        """
        while True:
            for batch_data in self._process_file_list(self._local_file_list):
                self._cache[batch_data._id] = batch_data
                # report and then get
                meta = self._data_client.get_batch_data_meta()
                if not self._set_batch_data(meta):
                    break
                continue

            break

        logger.info("local data process completed.")
        while True:
            meta = self._data_client.get_batch_data_meta()
            if not self._set_batch_data(meta):
                break

    @property
    def endpoint(self):
        return "{}:{}".format(utils.get_extern_ip(), self._serer.port)

    def get_id(self):
        return self._id

    @property
    def name(self):
        return self._name
