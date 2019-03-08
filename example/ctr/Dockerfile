FROM ubuntu:16.04
MAINTAINER peizhilin@baidu.com

RUN apt-get update && apt-get install -y python python-pip iputils-ping libgtk2.0-dev wget vim net-tools iftop python-opencv git curl

RUN pip install -U pip
RUN pip install -U kubernetes paddlepaddle
RUN mkdir -p /workspace

RUN mkdir -p /temp && cd /temp && git clone https://github.com/PaddlePaddle/models.git && cd models && git checkout f503908d && mv /temp/models/fluid/PaddleRec/ctr /workspace/

ADD script/paddle_k8s /usr/bin
ADD script/k8s_tools.py /root
RUN chmod +x /usr/bin/paddle_k8s

COPY ctr /workspace/ctr

