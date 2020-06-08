#!/bin/bash

set -eu

if [ ! -f ResNeXt101_32x16d_wsl_model.tar.gz ]; then
  wget --no-check-certificate https://paddle-edl.bj.bcebos.com/distill_teacher_model/ResNeXt101_32x16d_wsl_model.tar.gz
fi
tar -zxf ResNeXt101_32x16d_wsl_model.tar.gz

port=9898
python -m paddle_serving_server_gpu.serve \
  --model ResNeXt101_32x16d_wsl_model \
  --thread 4 \
  --port ${port} \
  --mem_optim True \
  --gpu_ids 1
