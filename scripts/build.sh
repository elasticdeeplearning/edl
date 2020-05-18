#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

pushd python/paddle_edl/protos/
bash generate.sh
popd

# TODO(gongwb):mv to devel image
python -m pip install  paddlepaddle-gpu==1.8.0.post107

# go
mkdir -p ${build_dir}/cmd/master/
go build   -o build/cmd/master/master cmd/master/master.go

#build python
build_dir=build
pushd ${build_dir}
cmake ..
make clean && make -j
ctest -V -R
popd

#test all go test
go test --cover ./...
