#!/bin/bash
set -e
if [[ $# != 1 ]] ; then
    echo "must set  python version"
    exit 0
fi

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

# check python version
which python
version_str=$(python --version 2>&1)
py_version=$1
if [[ ${version_str} != "Python ${py_verion}"* ]]; then
  echo "${version_str} not valid for argument:${py_version}"
  exit 1
fi


pushd python/paddle_edl/protos/
bash generate.sh
popd

# TODO(gongwb): add them on async training
# go
#build_dir=build
#rm -rf ${build_dir}
#mkdir -p ${build_dir}/cmd/master/
#go build   -o build/cmd/master/master cmd/master/master.go

#build python
pushd ${build_dir}
cmake .. -DPY_VERSION=${py_version}
make clean && make -j
ctest -V -R
popd

#test all go test
#go test --cover ./...
