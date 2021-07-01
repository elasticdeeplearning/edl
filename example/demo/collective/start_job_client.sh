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

set -e
unset http_proxy https_proxy

# running under edl
export PADDLE_RUNING_ENV=PADDLE_EDL
export PADDLE_JOBSERVER="http://127.0.0.1:8180"
if [[ "${PADDLE_TRAINERS}x" != x ]]; then
    pod_arr=(${PADDLE_TRAINERS//,/ })
    export PADDLE_JOBSERVER="http://${pod_arr[0]}:8180"
fi
export PADDLE_JOB_ID="test_job_id_1234"
export PADDLE_POD_ID="not set"

BASEDIR=$(dirname $(readlink -f $0))
echo $BASEDIR

python -m paddle_edl.demo.collective.job_client_demo \
    --log_level 20 \
    --package_sh ./resnet50/package.sh \
    --pod_path ./resnet50_pod \
    ./train_pretrain.sh
