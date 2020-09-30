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
BASEDIR=$(dirname "$(readlink -f "${0}")")


echo "base_dir:${BASEDIR}"
cd "${BASEDIR}"

# 2.7 is deprecated
# ./build.sh 2.7

function abort(){
    echo "Your change doesn't follow Edl's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}


function check_style() {
    trap 'abort' 0

    set +e
    upstream_url='https://github.com/elasticdeeplearning/edl'
    git remote remove upstream
    git remote add upstream $upstream_url
    set -e
    git fetch upstream develop

    pre-commit install
    changed_files="$(git diff --name-only upstream/develop)"
    echo "$changed_files" | xargs pre-commit run --files

    trap : 0
}

pushd "${BASEDIR}/../"
check_style
popd


./build.sh 3.7
