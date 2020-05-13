#!/bin/bash
set -e
image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7

cp ../scripts/download_etcd.sh .
docker build --network host . -t ${image} -f Dockerfile.runtime
rm -f download_etcd.sh

docker push ${image}
