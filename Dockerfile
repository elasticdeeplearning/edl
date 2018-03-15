FROM    golang:1.8
RUN     go get github.com/Masterminds/glide
RUN     apt-get update && apt-get install -y git
WORKDIR $GOPATH/src/github.com/paddlepaddle
RUN     mkdir -p $GOPATH/src/github.com/paddlepaddle/edl
# Add ENV http_proxy=[your proxy server] if needed
# run glide install before building go sources, so that
# if we change the code and rebuild the image can cost little time
ADD     ./glide.yaml ./glide.lock $GOPATH/src/github.com/paddlepaddle/edl/
WORKDIR $GOPATH/src/github.com/paddlepaddle/edl
RUN     glide install --strip-vendor
ADD     . $GOPATH/src/github.com/paddlepaddle/edl
RUN     go build -o /usr/local/bin/edl github.com/paddlepaddle/edl/cmd/edl
RUN     rm -rf $GOPATH/src/github.com/paddlepaddle/edl
CMD     ["edl"]
