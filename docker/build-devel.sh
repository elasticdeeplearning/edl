#!/bin/bash
set -e

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))
cd ${BASEDIR}/..

image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7-dev
sed 's/<baseimg>/latest-gpu-cuda10.0-cudnn7-dev/g' docker/Dockerfile > docker/Dockerfile.cuda10
docker build  --network host . -t ${image} -f docker/Dockerfile.cuda10
docker push ${image}

image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda9.0-cudnn7-dev
sed 's/<baseimg>/latest-gpu-cuda9.0-cudnn7-dev/g' docker/Dockerfile > docker/Dockerfile.cuda9
docker build  --network host . -t ${image} -f docker/Dockerfile.cuda9
docker push ${image}
