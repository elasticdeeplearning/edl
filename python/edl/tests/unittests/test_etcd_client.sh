#!/bin/bash
set -e

nohup etcd > test_etcd_client_etcd.log 2>&1 &
etcd_pid=$!

unset https_proxy http_proxy
python -u ./etcd_client_test.py

set +e
kill -9 $etcd_pid
echo $etcd_pid
