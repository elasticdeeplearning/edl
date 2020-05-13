FROM paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev

# Install Go
RUN rm -rf /usr/local/go && wget -qO- https://dl.google.com/go/go1.13.10.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir -p /root/gopath && \
    mkdir -p /root/gopath/bin && \
    mkdir -p /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.
ENV PATH=$PATH:{GOROOT}/bin:${GOPATH}/bin

# python
ADD ./docker/requirements.txt /root/paddle_edl/requirements.txt
RUN python -m pip install -r /root/paddle_edl/requirements.txt
#RUN python -m pip install paddlepaddle-gpu -i  http://mirrors.aliyun.com/pypi/simple/

#etcd
ENV HOME /root
WORKDIR /root/paddle_edl
ADD ./scripts/download_etcd.sh /root/paddle_edl/download_etcd.sh
RUN /root/paddle_edl/download_etcd.sh

# Install redis
RUN cd /tmp/ && wget http://download.redis.io/releases/redis-6.0.1.tar.gz &&  \
    tar xzf redis-6.0.1.tar.gz && \
    cd redis-6.0.1 && make -j && \
    mv src/redis-server /usr/local/bin && \
    mv src/redis-cli /usr/local/bin && \
    cd .. && rm -rf redis-6.0.1.tar.gz redis-6.0.1


