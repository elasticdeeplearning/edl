#!/bin/bash
set -e

nohup etcd 2>&1 &
pid=$!

unset https_proxy http_proxy
python ./etcd_client_test.py

kill -9 $pid
