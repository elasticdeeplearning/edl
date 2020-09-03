#!/bin/bash
set -e
unset https_proxy http_proxy

name=${TEST_TARGET_NAME}
TEST_TIMEOUT=${TEST_TIMEOUT}

# rm flag file
rm -f ${name}_*.log

nohup etcd > ${name}_etcd.log 2>&1 &
etcd_pid=$!

echo "etcd_pid:${etcd_pid} ${name}_etcd.log"

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

key="/${PADDLE_JOB_ID}/job_flag/nodes/complete"
value=`etcdctl get ${key}`
echo "job complete flag:${value}"

job_flag=True
for pid in $pid_00 $pid_01; do
    echo "wait ${pid}"
    if ! wait ${pid} ; then
        job_flag=False
    fi
done
#----

if [[ $job_flag == "True" ]]; then
    exit 0
fi

echo "cat ${name}_run_00.log"
cat ${name}_run_00.log

echo "cat ${name}_run_01.log"
cat ${name}_run_01.log


set +e
kill -9 $etcd_pid
echo $etcd_pid
exit 1
