#!/bin/bash
set -e

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))
cd ${BASEDIR}/..

image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7

docker build --network host . -t ${image} -f docker/Dockerfile.runtime
docker push ${image}
