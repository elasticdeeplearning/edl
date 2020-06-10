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
version_str=`python --version`
py_version=$1
if [[ $version_string != *"Python ${py_verion}"* ]]; then
  echo "${version_str} not valid for argument:${py_verion}"
fi


pushd python/paddle_edl/protos/
bash generate.sh
popd

# TODO(gongwb):mv to devel image
python${py_verion} -m pip install  paddlepaddle-gpu==1.8.0.post107

# go
build_dir=build
rm -rf ${build_dir}
mkdir -p ${build_dir}/cmd/master/
go build   -o build/cmd/master/master cmd/master/master.go

#build python
pushd ${build_dir}
cmake .. -DPY_VERION=${py_version}
exit 0
make clean && make -j
ctest -V -R
popd

#test all go test
go test --cover ./...
