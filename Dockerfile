FROM paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev

# Install Go
RUN wget -qO- https://dl.google.com/go/go1.13.10.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir /root/gopath && \
    mkdir /root/gopath/bin && \
    mkdir /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.
ENV PATH=PATH:{GOROOT}/bin:${GOPATH}/bin

RUN python -m pip install etcd3==0.12.0 grpcio_tools==1.28.1 grpcio==1.28.1 flask==1.1.2 pathlib==1.0.1
RUN python -m pip install paddlepaddle-gpu

ENV HOME /root
WORKDIR /root/paddle_edl
ADD ./scripts/download_etcd.sh /root/paddle_edl/download_etcd.sh

# Install redis
RUN cd /tmp/ && wget http://download.redis.io/releases/redis-6.0.1.tar.gz &&  \
    tar xzf redis-6.0.1.tar.gz && \
    cd redis-6.0.1 && make -j && \
    mv src/redis-server /usr/local/bin && \
    mv src/redis-cli /usr/local/bin && \
    cd .. && rm -rf redis-6.0.1.tar.gz redis-6.0.1


