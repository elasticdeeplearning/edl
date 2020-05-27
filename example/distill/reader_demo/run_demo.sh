#!/bin/bash

# paddle cloud will set discovery env, set for local test
export PADDLE_DISTILL_BALANCE_SERVER='10.255.100.13:9379'
export PADDLE_DISTILL_SERVICE_NAME=MnistDistill

python distill_reader_demo.py
