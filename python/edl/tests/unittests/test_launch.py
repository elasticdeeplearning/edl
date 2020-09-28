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

from __future__ import print_function

from edl.tests.unittests import etcd_test_base
from edl.utils import status as edl_status
from edl.utils.log_utils import logger
from edl.utils import launcher as edl_launcher


class TestLauncher(etcd_test_base.EtcdTestBase):
    def setUp(self):
        super(TestLauncher, self).setUp("test_launcher")

    def test_normal_exit(self):
        launcher = edl_launcher(self._job_env, self._pod, self._etcd, None)
        launcher.init()
        launcher.launch()

        last_status = edl_status.load_job_status_from_etcd(self._etcd)
        if last_status == edl_status.Status.SUCCEED:
            logger.info(
                "job:{} has completed! Need't try!".format(self._job_env.job_id)
            )
            return
        self.assertFalse(True)
