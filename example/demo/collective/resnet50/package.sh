#!/bin/bash
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
