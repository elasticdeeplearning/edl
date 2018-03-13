FROM    golang:1.8
RUN     go get github.com/Masterminds/glide
RUN     apt-get update && apt-get install -y git
WORKDIR $GOPATH/src/github.com/paddlepaddle
#RUN     git clone https://github.com/paddlepaddle/edl.git
RUN     mkdir -p $GOPATH/src/github.com/paddlepaddle/edl
ENV     http_proxy=http://172.19.32.166:8899
ENV     https_proxy=http://172.19.32.166:8899
ADD     ./glide.yaml ./glide.lock $GOPATH/src/github.com/paddlepaddle/edl/
WORKDIR $GOPATH/src/github.com/paddlepaddle/edl
RUN     glide install --strip-vendor
ADD     . $GOPATH/src/github.com/paddlepaddle/edl
RUN     go build -o /usr/local/bin/edl github.com/paddlepaddle/edl/cmd/edl
RUN     rm -rf $GOPATH/src/github.com/paddlepaddle/edl
CMD     ["edl"]
