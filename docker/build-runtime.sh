#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [[ $# != 1 ]] ; then
    echo "must set version"
    exit 0
fi

unset GREP_OPTIONS
BASEDIR="$(dirname "$(readlink -f "${0}")")"
cd "${BASEDIR}"/..

build_image(){
    cuda_version=$1
    latest_image="hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda${cuda_version}-cudnn7"
    sed 's/<baseimg>/1.8.0-gpu-cuda'"${cuda_version}"'-cudnn7/g' docker/Dockerfile.runtime > "docker/Dockerfile.runtime.cuda${cuda_version}"
    docker build --pull  --network host . -t "${latest_image}" -f "docker/Dockerfile.runtime.cuda${cuda_version}"
    docker push "${latest_image}"

    version=$2
    version_image="hub.baidubce.com/paddle-edl/paddle_edl:${version}-cuda${cuda_version}-cudnn7"
    docker tag "${latest_image}" "${version_image}"
    docker push "${version_image}"
}

version=$1
cuda_version="10.0"
echo "build cuda:${cuda_version} edl version:${version}"
build_image "${cuda_version}" "$version"

cuda_version="9.0"
echo "build cuda:${cuda_version} edl version:${version}"
build_image "${cuda_version}" "$version"
