#!/bin/bash
set -e
unset http_proxy https_proxy

# running under edl
export PADDLE_RUNING_ENV=PADDLE_EDL
export PADDLE_JOBSERVER="http://127.0.0.1:8180"
if [[ "${PADDLE_TRAINERS}x" != x ]]; then
    pod_arr=(${PADDLE_TRAINERS//,/ })
    export PADDLE_JOBSERVER="http://${pod_arr[0]}:8180"
fi
export PADDLE_JOB_ID="test_job_id_1234"
export PADDLE_POD_ID="not set"

BASEDIR=$(dirname $(readlink -f $0))
echo $BASEDIR

nohup python -u ${BASEDIR}/job_client_demo.py \
    --log_level 20 \
    --package_sh ./resnet50/package.sh \
    --pod_path ./resnet50_pod \
    ./train_pretrain.sh > job_client.log 2>&1 &
