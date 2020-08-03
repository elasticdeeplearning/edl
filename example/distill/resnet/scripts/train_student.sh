#!/bin/bash
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
