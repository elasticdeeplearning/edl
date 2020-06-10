#!/bin/bash
set -e

if [[ $# != 1 ]] ; then
    echo "must set version"
    exit 0
fi

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))
cd ${BASEDIR}/..

build_image(){
    cuda_version=$1
    latest_image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda${cuda_version}-cudnn7
    sed 's/<baseimg>/1.8.0-gpu-cuda'"${cuda_version}"'-cudnn7/g' docker/Dockerfile.runtime > docker/Dockerfile.runtime.cuda${cuda_version}
    docker build --network host . -t ${latest_image} -f docker/Dockerfile.runtime.cuda${cuda_version}
    docker push ${latest_image}

    version=$2
    version_image=hub.baidubce.com/paddle-edl/paddle_edl:${version}-cuda${cuda_version}-cudnn7
    docker tag ${latest_image} ${version_image}
    docker push ${version_image}
}

version=$1
cuda_version="10.0"
echo "build cuda:${cuda_version} edl version:${version}"
build_image ${cuda_version} $version

cuda_version="9.0"
echo "build cuda:${cuda_version} edl version:${version}"
build_image ${cuda_version} $version
