#!/bin/bash
set -e

if [[ $# != 1 ]] ; then
    echo "must set version"
    exit 0
fi

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))
cd ${BASEDIR}/..

latest_image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7

docker build --network host . -t ${latest_image} -f docker/Dockerfile.runtime
docker push ${latest_image}

version=$1

version_image=hub.baidubce.com/paddle-edl/paddle_edl:${version}-cuda10.0-cudnn7
docker tag ${latest_image} ${version_image}
docker push ${version_image}
