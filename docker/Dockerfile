FROM hub.baidubce.com/paddlepaddle/paddle:<baseimg>

# Install Go
RUN rm -rf /usr/local/go && wget -qO- https://dl.google.com/go/go1.13.10.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir -p /root/gopath && \
    mkdir -p /root/gopath/bin && \
    mkdir -p /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.
ENV PATH=$PATH:${GOROOT}/bin:${GOPATH}/bin

# python
ADD ./docker/requirements.txt /root/paddle_edl/requirements.txt
RUN python -m pip install pip==20.1.1
RUN python3.6 -m pip install pip==20.1.1
RUN python -m pip install -r /root/paddle_edl/requirements.txt
RUN python3.6 -m pip install -r /root/paddle_edl/requirements.txt

#etcd
ENV HOME /root
WORKDIR /root/paddle_edl
ADD ./scripts/download_etcd.sh /root/paddle_edl/download_etcd.sh
RUN bash /root/paddle_edl/download_etcd.sh

# Install redis
RUN cd /tmp/ && wget -q https://paddle-edl.bj.bcebos.com/redis-6.0.1.tar.gz &&  \
    tar xzf redis-6.0.1.tar.gz && \
    cd redis-6.0.1 && make -j && \
    mv src/redis-server /usr/local/bin && \
    mv src/redis-cli /usr/local/bin && \
    cd .. && rm -rf redis-6.0.1.tar.gz redis-6.0.1


# protoc
RUN mkdir -p /tmp/protoc && cd /tmp/protoc && \
    wget -q -O protoc-3.11.4-linux-x86_64.zip  --no-check-certificate  https://paddle-edl.bj.bcebos.com/protoc-3.11.4-linux-x86_64.zip && \
    unzip protoc-3.11.4-linux-x86_64.zip && mv bin/protoc /usr/local/bin

RUN echo "go env -w GO111MODULE=on && go env -w GOPROXY=https://goproxy.io,direct" >> /root/.bashrc
ENV GO111MODULE=on
ENV GOPROXY=https://goproxy.io,direct

