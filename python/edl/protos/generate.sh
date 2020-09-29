#!/bin/bash
set -xe

#TODO(gongwb): reopen them then async trainning
#pushd /tmp/
#go get -u -v github.com/golang/protobuf/protoc-gen-go@v1.3.0
#popd

#protoc --go_out=plugins=grpc:./  master.proto
#protoc --go_out=plugins=grpc:./  common.proto

#mkdir -p ../../../pkg/masterpb
#mv *.go ../../../pkg/masterpb

# see the build.sh to get the pakage version
which python
python ./run_codegen.py

# generate python compatabile path
sed -i -r 's/^import (.+_pb2.*)/from . import \1/g' ./*_pb2*.py

# import os
mv pod_server*.py data_server*.py common*.py ../utils/
mv distill_discovery*.py ../distill/
