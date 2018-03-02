FROM    golang:1.8
RUN     go get github.com/Masterminds/glide
RUN     apt-get update && apt-get install -y git
WORKDIR $GOPATH/src/github.com/paddlepaddle
RUN     git clone https://github.com/paddlepaddle/edl.git
WORKDIR $GOPATH/src/github.com/paddlepaddle/edl
RUN     glide install --strip-vendor
RUN     go build -o /usr/local/bin/edl github.com/paddlepaddle/edl/cmd/edl
CMD     ["edl"]
