#!/bin/bash

function delete_job() {
  jobname=$1
  kubectl delete trainingjob $jobname
  kubectl delete job $jobname-trainer
  kubectl delete rs $jobname-master $jobname-pserver
}

function delete_all() {
  jobs=$(kubectl get trainingjob | tail -n +2 | awk '{print $1}')
  for job in "${jobs[@]}"
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

