#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

echo ${BASEDIR}
cd ${BASEDIR}

./build.sh 2.7

python3.6 -m pip install pip==20.1.1
./build.sh 3.6
