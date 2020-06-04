#!/bin/bash

set -eu

# at gpu 0, start paddle serving server on port 9292
port=9292
nohup python -m paddle_serving_server_gpu.serve \
  --model mnist_model \
  --thread 4 \
  --port ${port} \
  --mem_optim True \
  --gpu_ids 0 &
serving_pid=$!

# start distill train
export CUDA_VISIBLE_DEVICES=0
python train_with_fleet.py \
  --use_distill_service True \
  --distill_teachers 127.0.0.1:${port}

kill -9 ${serving_pid}
