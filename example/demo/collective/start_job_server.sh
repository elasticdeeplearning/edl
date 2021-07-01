#!/bin/bash

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

node_ips="127.0.0.1"
if [[ "${PADDLE_TRAINERS}x" != "x" ]]; then
    node_ips=${PADDLE_TRAINERS}
fi
echo "node_ips:${node_ips}"

BASEDIR=$(dirname $(readlink -f $0))
echo "${BASEDIR}"

python -m paddle_edl.demo.collective.job_server_demo \
    --node_ips ${node_ips} \
    --pod_num_of_node 8 \
    --time_interval_to_change 900 \
    --gpu_num_of_node 8
