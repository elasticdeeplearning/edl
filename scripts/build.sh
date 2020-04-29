#!/bin/bash
set -xe
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

pip install etcd3 grpcio_tools grpcio flask pathlib --ignore-installed
pip install paddlepaddle-gpu  --ignore-installed

pushd python/paddle_edl/protos/
bash generate.sh
popd


build_dir=build
mkdir -p  ${build_dir}
cd ${build_dir}

cmake ..
make clean && make -j
ctest -V -R
