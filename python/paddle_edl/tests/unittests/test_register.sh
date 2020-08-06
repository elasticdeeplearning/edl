#!/bin/bash
set -e

export PADDLE_ETCD_ENPOINTS="127.0.0.1:2379"
export CUDA_VISIBLE_DEVICES=0

name=${TEST_TARGET_NAME}
TEST_TIMEOUT=${TEST_TIMEOUT}

if [[ ${TEST_TIMEOUT}"x" == "x" ]]; then
    echo "can't find ${TEST_TIMEOUT}, please set ${TEST_TIMEOUT} first"
    exit 1
fi

# rm flag file
rm -f ${name}_*.log

nohup etcd > ${name}_etcd.log 2>&1 &
etcd_pid=$!

# start the unit test
run_time=$(( $TEST_TIMEOUT - 10 ))
echo "run_time: ${run_time}"

timeout -s SIGKILL ${run_time} python register_test.py > ${name}_run.log 2>&1

kill -9 $etcd_pid
