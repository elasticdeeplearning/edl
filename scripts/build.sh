#!/bin/bash
set -e
if [[ $# != 1 ]] ; then
    echo "must set  python version"
    exit 0
fi

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

function gen_env(){
    py_version=$1
    old_path=$PATH
    python_path=`which python${py_version}`

    tmp_path=/tmp/edl-build/python${py_version}/bin
    mkdir -p  ${tmp_path}
    rm -f ${tmp_path}/python

    ln -s ${python_path} ${tmp_path}/python
    export PATH=${tmp_path}:$old_path
    echo "current path:${PATH}"
}

py_version=$1
gen_env $py_version

# check python version
which python
version_str=$(python --version 2>&1)
echo "python version:${version_str}"
if [[ ${version_str} != "Python ${py_verion}"* ]]; then
  echo "${version_str} not valid for argument:${py_version}"
  exit 1
fi


pushd python/edl/protos/
bash generate.sh
popd

build_dir=build
rm -rf ${build_dir}
mkdir -p ${build_dir}/cmd/master/
# TODO(gongwb): add them on async training
# go
#go build   -o build/cmd/master/master cmd/master/master.go

#build python
pushd ${build_dir}
cmake .. -DPY_VERSION=${py_version}
make clean && make -j
ctest -V -R
popd

#test all go test
#go test --cover ./...
