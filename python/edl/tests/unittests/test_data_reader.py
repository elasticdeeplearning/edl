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

import unittest
import os

from edl.collective import distribute_reader
from edl.collective import dataset
from edl.tests.unittests import etcd_trainer_test_base
from edl.collective import state as edl_state


class Args(object):
    def __init__(self):
        self.job_id = None
        self.pod_id = None
        self.global_rank = None
        self.rank_in_pod = None
        self.trainer_endpoints = None
        self.pod_ids = None
        self.gpu_id = "0"


class TestDataReader(etcd_trainer_test_base.EtcdTestBase):
    def _init_args(self, pod_id, global_rank, rank_in_pod):
        args = etcd_trainer_test_base.Args()
        self.job_id = self._job_id
        self.pod_id = str(pod_id)
        self.global_rank = str(global_rank)
        self.rank_in_pod = str(rank_in_pod)
        self.trainer_endpoints = None
        self.pod_ids = "0,1"
        self.gpu_id = "0"

        return args

    def _update_env(self, pod_id, global_rank, rank_in_pod):
        args = self._init_args(pod_id, global_rank, rank_in_pod)
        proc_env = {
            "PADDLE_JOB_ID": args.job_id,
            "PADDLE_POD_ID": args.pod_id,
            "EDL_POD_LEADER_ID": "0",
            "PADDLE_ETCD_ENDPOINTS": "127.0.0.1:2379",
            "PADDLE_TRAINER_ID": args.global_rank,
            "PADDLE_TRAINER_RANK_IN_POD": args.rank_in_pod,
            "EDL_POD_IDS": args.pod_ids,
            "PADDLE_TRAINER_ENDPOINTS": args.trainer_endpoints,
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_PATH": "hdfs://{}".format(args.job_id),
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".{}".format(args.job_id),
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0",
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
        }
        os.environ.pop("https_proxy", None)
        os.environ.pop("http_proxy", None)
        os.environ.update(proc_env)

    def setUp(self):
        self._job_id = "test_data_reader"
        super(TestDataReader, self).setUp(self._job_id)

        self._file_list = ["./data_server/a.txt", "./data_server/b.txt"]
        self._data = {}
        for idx, p in enumerate(self._file_list):
            reader = dataset.TxtFileSplitter(p)
            for rec in reader:
                if idx not in self._data:
                    self._data[idx] = []
                self._data[idx].append(rec)

    def test_data_reader(self):
        self._update_env(pod_id="0", global_rank=0, rank_in_pod=0)
        state = edl_state.PaddleState(total_batch_size=1)
        reader1 = distribute_reader.Reader(
            state=state,
            file_list=self._file_list,
            file_splitter_cls=dataset.TxtFileSplitter,
            batch_size=1,
        )

        self._update_env(pod_id="1", global_rank=1, rank_in_pod=0)
        state = edl_state.PaddleState(total_batch_size=1)
        reader2 = distribute_reader.Reader(
            state=state,
            file_list=self._file_list,
            file_splitter_cls=dataset.TxtFileSplitter,
            batch_size=1,
        )

        size1 = 0
        for meta, batch in reader1:
            self.assertTrue(meta._size, 1)
            for k, v in meta._batch:
                c = self._data[k._idx]
                self.assertTrue(c[0][0], k._path)
                size1 += 1

        size2 = 0
        for meta, batch in reader2:
            self.assertTrue(meta._size, 1)
            for k, v in meta._batch:
                c = self._data[k._idx]
                self.assertTrue(c[0][0], k._path)
                size2 += 1

        self.assertTrue(size1, size2)


if __name__ == "__main__":
    unittest.main()
