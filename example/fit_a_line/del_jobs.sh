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

function delete_job() {
  jobname=$1
  if [[ "$jobname" == "" ]]; then
      echo "Usage: sh edl_jobs.sh [all|<job-name>]"
      exit 0
  fi
  kubectl delete trainingjob $jobname
  kubectl delete job $jobname-trainer
  kubectl delete rs $jobname-master $jobname-pserver
}

function delete_all() {
  jobs=$(kubectl get trainingjob | tail -n +2 | awk '{print $1}')
  for job in ${jobs[@]}
  do
    delete_job $job
  done
}

case "$1" in
    all)
      delete_all
      ;;
    *)
      delete_job $1
      ;;
esac
