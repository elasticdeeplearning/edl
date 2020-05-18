#!/bin/bash
set -xe

pushd /tmp/ 
go get -u -v github.com/golang/protobuf/protoc-gen-go@v1.3.0
popd

protoc --go_out=plugins=grpc:./  master.proto
protoc --go_out=plugins=grpc:./  common.proto

mkdir -p ../../../pkg/masterpb
mv *.go ../../../pkg/masterpb
# see the build.sh to get the pakage version
python ./run_codegen.py
 
