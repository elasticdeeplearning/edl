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

set -eu

if [ ! -f mnist_cnn_model.tar.gz ]; then
  wget --no-check-certificate https://paddle-edl.bj.bcebos.com/distill_teacher_model/mnist_cnn_model.tar.gz
fi
tar -zxf mnist_cnn_model.tar.gz

# at gpu 0, start paddle serving server on port 9292
port=9292
nohup python -m paddle_serving_server_gpu.serve \
  --model mnist_cnn_model \
  --thread 4 \
  --port ${port} \
  --mem_optim True \
  --gpu_ids 0 &
serving_pid=$!

python distill_reader_demo.py --distill_teachers 127.0.0.1:${port}

# kill serving server
pstree -p ${serving_pid} | awk -F"[()]" '{print $2}'| xargs kill -9
