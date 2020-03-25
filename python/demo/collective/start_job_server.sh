#!/bin/bash
node_ips="127.0.0.1"
if [[ "${PADDLE_TRAINERS}x" != "x" ]]; then
    node_ips=${PADDLE_TRAINERS}
fi
echo "node_ips:${node_ips}"

BASEDIR=$(dirname $(readlink -f $0))
echo "${BASEDIR}"

nohup python -u ${BASEDIR}/job_server_demo.py \
    --node_ips ${node_ips} \
    --pod_num_of_node 2 \
    --time_interval_to_change 900 \
    --gpu_num_of_node 8 > job_server.log 2>&1 &
