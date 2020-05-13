#!/bin/bash
set -e
image=hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7-example-demo

docker build --network host . -t ${image} -f docker/Dockerfile.example-demo
docker push ${image}
