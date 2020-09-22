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

from edl.utils import constants
from edl.utils import register
from edl.utils import error_utils
from edl.utils import pod
from edl.utils import string_utils
from edl.utils import exceptions


class PodResourceRegister(register.Register):
    def __init__(self, job_env, pod_id, pod_json, ttl=constants.ETCD_TTL):
        service = constants.ETCD_POD_RESOURCE
        server = "{}".format(pod_id)
        value = pod_json

        super(PodResourceRegister, self).__init__(
            etcd_endpoints=job_env.etcd_endpoints,
            job_id=job_env.job_id,
            service=service,
            server=server,
            info=value,
            ttl=ttl)


@error_utils.handle_errors_until_timeout
def load_from_etcd(etcd, timeout=15):
    servers = etcd.get_service(constants.ETCD_POD_RESOURCE)

    pods = {}
    for s in servers:
        p = pod.Pod()
        p.from_json(string_utils.bytes_to_string(s.info))
        pods[p.get_id()] = p

    return pods


@error_utils.handle_errors_until_timeout
def wait_resource(pod_id, timeout=15):
    pods = load_from_etcd(timeout=timeout)
    if len(pods) == 1:
        if pod_id in pods:
            return True

    if len(pods) == 0:
        return True

    raise exceptions.EdlWaitFollowersReleaseError(
        "can't wait all resource exit:{}".format(pods.keys()))

    return False
