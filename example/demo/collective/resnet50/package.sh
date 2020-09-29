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

set -xe

while true ; do
  case "$1" in
    -pod_id) pod_id="$2" ; shift 2 ;;
    *)
       if [[ ${#1} -gt 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi
       ;;
  esac
done


src_dir=../../../collective/resnet50
dst_dir=resnet50_pod/${pod_id}

echo "mkdir resnet50_pod/${pod_id}"
mkdir -p  "${dst_dir}"

#copy resnet50 runtime env
cp "${src_dir}"/*.py  "${dst_dir}"/
cp "${src_dir}"/*.sh "${dst_dir}"/
cp -r "${src_dir}"/utils "${dst_dir}"/utils
cp -r "${src_dir}"/models "${dst_dir}"/models
cp -r "${src_dir}"/scripts "${dst_dir}"/scripts

if [[ ! -d "${dst_dir}/ImageNet" ]]; then
    ln -s "${PADDLE_EDL_IMAGENET_PATH}" "${dst_dir}"/
fi

if [[ ! -d "${dst_dir}/fleet_checkpoints" ]]; then
    ln -s "${PADDLE_EDL_FLEET_CHECKPOINT_PATH}" "${dst_dir}/fleet_checkpoints"
fi
