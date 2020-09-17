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

import os

import unittest
from edl.utils import data_server
from edl.utils import data_server_client
from edl.utils import edl_env
from edl.utils import log_utils
from edl.utils.log_utils import logger


class TestDataServer(unittest.TestCase):
    def setUp(self):
        self._job_id = "test_data_server_job_id"
        self._pod_id = "test_data_server_pod_id"
        self._reader_name = "test_data_server_reader"
        self._file_list = ["./data_server/a.txt", "./data_server/b.txt"]

        self._old_environ = dict(os.environ)

        proc_env = {
            "PADDLE_JOB_ID": self._job_id,
            "PADDLE_POD_ID": self._pod_id,
            "PADDLE_ETCD_ENDPOINTS": "127.0.0.1:2379",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINER_RANK_IN_POD": "0",
            "PADDLE_TRAINER_ENDPOINTS": "127.0.0.1:0",
            "EDL_POD_IDS": self._pod_id,
        }

        os.environ.update(proc_env)

        self._trainer_env = edl_env.TrainerEnv()
        self._data_server = data_server.DataServer(
            trainer_env=self._trainer_env,
            reader_name=self._reader_name,
            file_list=self._file_list,
            local_reader=None)
        self._data_server.start(addr="127.0.0.1")
        logger.info("start data server:{}".format(self._data_server.endpoint))

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)

    def test_get_file_list(self):
        client = data_server_client.Client()
        file_list = client.get_file_list(
            self._data_server.endpoint,
            self._reader_name,
            self._pod_id,
            self._file_list,
            timeout=60)
        self.assertEqual(len(file_list), len(self._file_list))

        for ele in file_list:
            self.assertEqual(ele.path, self._file_list[ele.idx])

    def test_batch_data_meta(self):
        client = data_server_client.Client()
        client.report_batch_data_meta(
            reader_leader_endpoint=self._data_server.endpoint,
            reader_name=self._reader_name,
            pod_id=self._pod_id,
            dataserver_endpoint=self._data_server.endpoint,
            batch_data_ids=["0", "1"],
            timeout=60)
        data = client.get_batch_data_meta(
            reader_leader_endpoint=self._data_server.endpoint,
            reader_name=self._reader_name,
            pod_id=self._pod_id,
            timeout=60)

        self.assertEqual(len(data), 1)
        self.assertEqual(len(data[0].batch_data_ids), 2)
        for batch_data_id in data[0].batch_data_ids:
            self.assertTrue(batch_data_id in ["0", "1"])

    #TODO(gongwb): add test_get_batch_data
    """
    def test_get_batch_data(self):
        pass
    """


if __name__ == '__main__':
    log_utils.get_logger(log_level=10)
    unittest.main()
