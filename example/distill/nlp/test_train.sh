#!/bin/bash
export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export CUDA_VISIBLE_DEVICES=7
nohup python3.6 -u train.py --model CNN > train_with_test.log 2>&1 &
