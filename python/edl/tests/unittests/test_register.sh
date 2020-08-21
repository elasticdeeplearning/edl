#!/bin/bash
set -e
unset https_proxy http_proxy

name=${TEST_TARGET_NAME}
TEST_TIMEOUT=${TEST_TIMEOUT}

nohup etcd > ${name}_etcd.log 2>&1 &
etcd_pid=$!


if [[ ${TEST_TIMEOUT}"x" == "x" ]]; then
    echo "can't find ${TEST_TIMEOUT}, please set ${TEST_TIMEOUT} first"
    exit 1
fi

# start the unit test
run_time=$(( $TEST_TIMEOUT - 10 ))
echo "run_time: ${run_time}"

export PADDLE_JOB_ID="test_success_job"
export PADDLE_ETCD_ENDPOINTS="127.0.0.1:2379"
export PADDLE_EDLNODES_RANAGE="2:2"
export PADDLE_EDL_ONLY_FOR_CE_TEST="1"
export PADDLE_EDL_HDFS_CHECKPOINT_PATH="./success_job"
export PADDLE_EDL_HDFS_HOME="./hadoop"

# rm flag file
rm -f ${name}_*.log
#clean keys
python del_from_etcd.py

# all success----
export CUDA_VISIBLE_DEVICES=0
export PADDLE_DEMO_EXIT_CODE=0
timeout -s SIGKILL ${run_time} python -m edl.collective.launch --log_dir 00 launch_demo.py > ${name}_run_00.log 2>&1 &
pid_00=$!

export CUDA_VISIBLE_DEVICES=1
export PADDLE_DEMO_EXIT_CODE=0
timeout -s SIGKILL ${run_time} python -m edl.collective.launch --log_dir 01 launch_demo.py > ${name}_run_01.log 2>&1 &
pid_01=$!

wait $pid_00 $pid_01

value=`etcdctl get ${key}`
echo $value
#----

kill -9 $etcd_pid
