#!/bin/bash
set -xe

protoc --go_out=plugins=grpc:./  master.proto
protoc --go_out=plugins=grpc:./  common.proto

mkdir -p ../../../pkg/masterpb
mv *.go ../../../pkg/masterpb
# see the build.sh to get the pakage version
python ./run_codegen.py
 
