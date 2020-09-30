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

RED='\033[0;31m'
NC='\033[0m' # No Color
GREEN='\033[0;32m'

if [[ $# != 1 ]] ; then
    echo "must set version"
    exit 0
fi

version=$1

echo -e "${GREEN} Press 'y' to release ${RED} docker version ${version} ${NC}"
while : ; do
    read -n 1 k <&1
    if [[ $k == y ]] ; then
        break
    else
        echo "exit"
        exit 0
    fi
done

echo -e "\n${GREEN} Begin to release ${RED} edl docker ${version} ${NC}\n"

unset GREP_OPTIONS
BASEDIR="$(dirname "$(readlink -f "${0}")")"
cd "${BASEDIR}"

bash ./build-runtime.sh "$version"
