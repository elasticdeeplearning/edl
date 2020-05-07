#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

python -m pip install etcd3 grpcio_tools grpcio flask pathlib --ignore-installed
python -m pip install paddlepaddle-gpu  --ignore-installed

pushd python/paddle_edl/protos/
bash generate.sh
popd


build_dir=build
mkdir -p  ${build_dir}
pushd ${build_dir}

cmake ..
make clean && make -j
unset http_proxy https_proxy
ctest -V -R

popd

#test all go test
go test --cover ./...
mkdir -p build/cmd/master/

#build
go build   -o build/master/master cmd/master/master.go
