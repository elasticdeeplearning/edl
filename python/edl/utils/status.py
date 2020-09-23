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

import enum
import json
from edl.utils import constants
from edl.utils import error_utils
from edl.utils.log_utils import logger


class Status(enum.IntEnum):
    INITIAL = 0
    RUNNING = 1
    PENDING = 2
    SUCCEED = 3
    FAILED = 4


def bool_to_status(b):
    if b:
        return Status.SUCCEED

    return Status.FAILED


@error_utils.handle_errors_until_timeout
def load_job_status_from_etcd(etcd, timeout=15):
    service = constants.ETCD_JOB_STATUS
    servers = etcd.get_service(service)

    assert len(servers) <= 1
    if len(servers) < 1:
        return None

    s = servers[0]
    d = json.loads(s.info)
    return d["status"]


@error_utils.handle_errors_until_timeout
def save_job_status_to_etcd(etcd, status, timeout=15):
    service = constants.ETCD_JOB_STATUS
    server = "status"
    info = json.dumps({"status": int(status)})
    etcd.set_server_permanent(service, server, info)


@error_utils.handle_errors_until_timeout
def save_job_flag_to_etcd(self, pod_id, flag, timeout=15):
    if flag:
        save_job_status_to_etcd(pod_id, Status.SUCCEED)
        logger.info("This job succeeded!")
        return

    logger.fatal("This job meets error!")


@error_utils.handle_errors_until_timeout
def save_pod_status_to_etcd(etcd, pod_id, status, timeout=15):
    service = constants.ETCD_POD_STATUS
    server = pod_id
    info = json.dumps({"status": int(status)})

    etcd.set_server_permanent(service, server, info)


@error_utils.handle_errors_until_timeout
def load_pods_status_from_etcd(etcd, timeout=15):
    service = constants.ETCD_POD_STATUS
    servers = etcd.get_service(service)

    inited = set()
    running = set()
    succeed = set()
    failed = set()
    for server in servers:
        d = json.loads(server.info)
        status = d["status"]
        if status == int(Status.FAILED):
            failed.add(server.server)
        elif status == int(Status.SUCCEED):
            succeed.add(server.server)
        elif status == int(Status.INITIAL):
            inited.add(server.server)
        elif status == int(Status.RUNNING):
            running.add(server.server)

    return inited, running, succeed, failed


@error_utils.handle_errors_until_timeout
def save_pod_flag_to_etcd(etcd, pod_id, flag, timeout=15):
    if not flag:
        save_pod_status_to_etcd(etcd, pod_id, Status.FAILED)
        logger.fatal("local trainers meets error!")
        return

    save_pod_status_to_etcd(etcd, pod_id, Status.SUCCEED, timeout=timeout)
    logger.info("local trainers succeeded!")
