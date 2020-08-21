#!/bin/bash
set -e

nohup etcd > test_etcd_client_etcd.log 2>&1 &
pid=$!

unset https_proxy http_proxy
python -u ./etcd_client_test.py

kill -9 $pid
