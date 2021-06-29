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

export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export CUDA_VISIBLE_DEVICES=7
nohup python3.6 -u train.py --model CNN > train_with_test.log 2>&1 &
