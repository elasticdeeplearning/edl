#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

python -m pip install etcd3==0.12.0 grpcio_tools==1.28.1 grpcio==1.28.1 flaski==1.1.2 pathlib==1.0.1 --ignore-installed
python -m pip install paddlepaddle-gpu  --ignore-installed

./scripts/download_etcd.sh

pushd python/paddle_edl/protos/
bash generate.sh
popd

#build master go
mkdir -p build/cmd/master/
go build   -o build/master/master cmd/master/master.go

#test all go test
go test --cover ./...

#build python
build_dir=build
mkdir -p  ${build_dir}
pushd ${build_dir}
cmake ..
make clean && make -j
ctest -V -R
popd
