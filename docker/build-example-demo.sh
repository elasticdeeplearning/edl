#!/bin/bash
set -e
image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7-exmaple-demo

docker build --network host . -t ${image} -f Dockerfile
docker push ${image}
