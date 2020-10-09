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

unset GREP_OPTIONS
BASEDIR="$(dirname "$(readlink -f "${0}")")"
cd "${BASEDIR}"/..

image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7-dev
sed 's/<baseimg>/latest-gpu-cuda10.0-cudnn7-dev/g' docker/Dockerfile > docker/Dockerfile.cuda10
docker build --pull  --network host . -t ${image} -f docker/Dockerfile.cuda10
docker push ${image}

image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda9.0-cudnn7-dev
sed 's/<baseimg>/latest-gpu-cuda9.0-cudnn7-dev/g' docker/Dockerfile > docker/Dockerfile.cuda9
docker build --pull  --network host . -t ${image} -f docker/Dockerfile.cuda9
docker push ${image}
