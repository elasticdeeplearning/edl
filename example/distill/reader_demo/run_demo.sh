#!/bin/bash

# at gpu 0, start paddle serving server on port 9292
port=9292
nohup python -m paddle_serving_server_gpu.serve \
  --model mnist_model \
  --thread 4 \
  --port ${port} \
  --mem_optim True \
  --gpu_ids 0 &
serving_pid=$!

python distill_reader_demo.py
kill -9 ${serving_pid}
