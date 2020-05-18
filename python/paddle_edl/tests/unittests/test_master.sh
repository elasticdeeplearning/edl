#!/bin/bash
set -e
nohup etcd > test_master_etcd.log 2>&1 &
etcd_pid=$!

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

nohup ${BASEDIR}/../../../../cmd/master 2>&1 &
master_pid=$!

unset https_proxy http_proxy
sleep 15s
python ./master_client_test.py

kill -9 $etcd_pid $master_pid
