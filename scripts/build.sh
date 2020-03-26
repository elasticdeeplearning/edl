#!/bin/bash
set -xe
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))
branch=`git branch | grep \* | cut -d ' ' -f2`

cd ${BASEDIR}/..
build_dir=build/build_${branch}
mkdir -p  ${build_dir}
cd ${build_dir}

cmake ../../
make -j
ctest -V -R test_*
