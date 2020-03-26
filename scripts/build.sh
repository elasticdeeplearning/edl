#!/bin/bash
set -xe
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..
build_dir=build
mkdir -p  ${build_dir}
cd ${build_dir}

cmake ..
make -j
ctest -V -R test_*
