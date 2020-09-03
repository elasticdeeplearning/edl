#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}

build_edl 2.7

python3.6 -m pip install pip==20.1.1
build_edl 3.6
