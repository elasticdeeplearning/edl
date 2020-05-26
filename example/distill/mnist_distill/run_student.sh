#!/bin/bash

set -eu

# paddle cloud will set discovery env, set for local test
export PADDLE_DISTILL_BALANCE_SERVER='10.255.100.13:9379'
#export PADDLE_DISTILL_BALANCE_SERVER='127.0.0.1:9379'
export PADDLE_DISTILL_SERVICE_NAME=MnistDistill

CUDA_VISIBLE_DEVICES=0 python train_student_with_fleet.py
