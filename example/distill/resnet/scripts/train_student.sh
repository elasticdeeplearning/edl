#!/bin/bash
export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_exhaustive_search=1
#export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_batchnorm_spatial_persistent=1

export GLOG_v=1
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO
# Unset proxy
unset https_proxy http_proxy

FP16=False #whether to use float16
use_dali=False
DATA_FORMAT="NCHW"
distributed_args="--selected_gpus 0"

model=ResNet50_vd

python -m paddle.distributed.launch ${distributed_args} \
       --log_dir log \
       ./train_with_fleet.py \
       --model=${model} \
       --batch_size=32 \
       --total_images=1281167 \
       --data_dir=./ImageNet \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=False \
       --lr_strategy=piecewise_decay \
       --lr=0.1\
       --l2_decay=1e-4 \
       --scale_loss=1.0 \
       --num_epochs=90 \
       --num_threads=2 \
       --nccl_comm_num=1 \
       --fuse=True \
       --use_hierarchical_allreduce=False \
       --fp16=${FP16} \
       --use_dali=${use_dali} \
       --do_test=False \
       --data_format=${DATA_FORMAT} \
       --use_distill_service=True \
       --teachers=127.0.0.1:9898
