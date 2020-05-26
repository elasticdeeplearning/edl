#!/bin/bash
unset https_proxy http_proxy

#nohup /tmp/etcd-download-test/etcd 2>&1 &
nohup etcd 2>&1 &
etcd_pid=$!

# wait etcd start
sleep 10

nohup python -m paddle_edl.discovery.register --service_name DistillReaderTest --server 127.0.0.1:2379 &
register_pid=$!

nohup python -m paddle_edl.distill.discovery_server &
discovery_pid=$!

export PADDLE_DISTILL_BALANCE_SERVER=127.0.0.1:7001
export PADDLE_DISTILL_SERVICE_NAME=DistillReaderTest
export PADDLE_DISTILL_MAX_TEACHER=4
python distill_reader_test.py

kill -9 $discovery_pid $register_pid $etcd_pid
