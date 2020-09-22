# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
paddle.distributed.launch is a module that spawns multiple distributed 
process on each training node for gpu training.
"""

from __future__ import print_function

import sys
from edl.utils import args_utils
from edl.utils import env as edl_env
from edl.utils import etcd_db
from edl.utils import launcher as edl_launcher
from edl.utils import log_utils
from edl.utils import status as edl_status
from edl.utils.logg_utils import logger
from edl.utils import pod as edl_pod

def main():
    log_utils.get_logger(log_level=10)
    args = args_utils.parse_args()
    args_dict = args_utils.convert_args_to_dict(args)

    # job enviroment.
    job_env = edl_env.JobEnv(args_dict)
    logger.info("get job env:{}".format(str(job_env)))

    # get global etcd and lock
    etcd = etcd_db.get_global_etcd(job_env.etcd_endpoints, job_env.job_id)

    last_status = edl_status.load_job_status_from_etcd(etcd)
    if last_status == edl_status.Status.SUCCEED:
        logger.info("job:{} has completed! Need't try!".format(job_env.job_id))
        sys.exit(0)

    # local pod, and the pod's id does't change.
    pod = edl_pod.Pod()
    pod.from_env(job_env)

    launcher = edl_launcher(job_env, pod, etcd, args)
    launcher.init()
    launcher.launch()

if __name__ == '__main__':
    main()
