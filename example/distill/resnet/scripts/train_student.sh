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

# Unset proxy
unset https_proxy http_proxy

export GLOG_v=1
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO

python -m paddle.distributed.launch --selected_gpus 0 \
       ./train_with_fleet.py \
       --model=ResNet50_vd \
       --data_dir=./ImageNet \
       --lr_strategy=cosine_warmup_decay \
       --use_distill_service=True \
       --distill_teachers=127.0.0.1:9898
