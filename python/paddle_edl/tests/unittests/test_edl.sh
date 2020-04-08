#!/bin/bash
#set -x

echo "python_path:${PYTHONPATH}"
unset http_proxy https_proxy

# start job_server
BASEDIR=$(dirname $(readlink -f $0))
echo "${BASEDIR}"

rm -rf job_server.log job_client.log ./edl_demo_log

nohup python -m paddle_edl.demo.collective.job_server_demo --pod_num_of_node 2 \
    --time_interval_to_change 900 \
    --gpu_num_of_node 2 \
    --pod_num_of_node 2 \
    --server_port 8180 > job_server.log 2>&1 &

server_pid=$!
echo "server_pid:${server_pid}"

# start job_client
# running under edl
export PADDLE_RUNING_ENV=PADDLE_EDL
export PADDLE_JOBSERVER="http://127.0.0.1:8180"
export PADDLE_JOB_ID="test_job_id_1234"
export PADDLE_POD_ID="not set"

nohup python -m paddle_edl.demo.collective.job_client_demo \
    --log_level 20 \
    --log_dir ./edl_demo_log \
    ./start_edl_demo.sh > job_client.log 2>&1 &

job_client_pid=$!
echo "launcher_pid:${job_client_pid}"
sleep 30s

echo "test request and response"
str="pod_0_0__edl_demo__"
file=./edl_demo_log/pod_pod_0_0.log

kill ${server_pid} ${job_client_pid}

if grep -q "$str" "$file"; then
    echo "request and response ok!"
else
    echo "request and response error!"
    echo "job_server.log"
    cat job_server.log
    echo "job_client.log"
    cat job_client.log
    exit -1
fi
