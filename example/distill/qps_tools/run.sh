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

# local test qps
export PADDLE_DISTILL_BALANCE_SERVER='10.255.100.13:9379'
export PADDLE_DISTILL_SERVICE_NAME=MnistDistill
export PADDLE_DISTILL_MAX_TEACHER=1
export PADDLE_DISTILL_CONF_FILE="$PWD/../reader_demo/serving_conf/serving_client_conf.prototxt"

batch_size=(1 2 4 8 16 24 32)
for x in ${batch_size[@]}; do
  echo "-------- batch_size=$x ---------"
  python distill_reader_qps.py --teacher_bs $x
  echo
done
