#!/bin/bash
set -xe
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

pushd python/paddle_edl/protos/
python run_codegen.py
popd

build_dir=build
mkdir -p  ${build_dir}
cd ${build_dir}

cmake ..
make clean && make -j
python -m pip install paddlepaddle-gpu
ctest -V -R
