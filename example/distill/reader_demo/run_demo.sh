#!/bin/bash

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
