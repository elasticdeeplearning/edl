#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}

function build_edl(){
    py_version=$1
    old_path=$PATH
    python_path=`which python${py_version}`

    tmp_path=/tmp/edl-build/python${py_version}/bin
    mkdir -p  ${tmp_path}
    rm -f ${tmp_path}/python

    ln -s ${python_path} ${tmp_path}/python
    export PATH=${tmp_path}:$old_path

    ./build.sh ${py_version}

    export PATH=$old_path
}

#build_edl 2.7

python3.6 -m pip install pip==20.1.1
build_edl 3.6
