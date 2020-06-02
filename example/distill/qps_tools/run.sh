#!/bin/bash

# local test qps
export PADDLE_DISTILL_BALANCE_SERVER='10.255.100.13:9379'
export PADDLE_DISTILL_SERVICE_NAME=MnistDistill
export PADDLE_DISTILL_MAX_TEACHER=1
export PADDLE_DISTILL_CONF_FILE="$PWD/../reader_demo/serving_conf/serving_client_conf.prototxt"

batch_size=(1 2 4 8 16 24 32)
for x in ${batch_size[@]}; do
  echo "-------- batch_size=$x ---------"
  python distill_reader_qps.py --teacher_bs $x
  echo
done
