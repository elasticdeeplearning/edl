#!/bin/bash
set -xe

if [[ ! -d "/tmp/protoc" ]]; then
    mkdir -p /tmp/protoc 
    pushd /tmp/protoc
    wget -O protoc-3.11.4-linux-x86_64.zip  --no-check-certificate  https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
    unzip protoc-3.11.4-linux-x86_64.zip
    popd
fi

pushd  /tmp/
go get -u -v github.com/golang/protobuf/protoc-gen-go@v1.3.0
popd
export PATH=/tmp/protoc/bin:$PATH:$(go env GOPATH)

protoc --go_out=plugins=grpc:./  master.proto
protoc --go_out=plugins=grpc:./  common.proto

mkdir -p ../../../pkg/masterpb
mv *.go ../../../pkg/masterpb
python ./run_codegen.py
 
