#!/bin/bash
nohup /tmp/etcd-download-test/etcd 2>&1 &
etcd_pid=$!

nohup ../../../../build/master/master 2>&1 &
master_pid=$!

unset https_proxy http_proxy
python ./master_client_test.py

kill -9 $etcd_pid $master_pid
