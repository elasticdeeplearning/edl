#!/bin/bash 
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

./build.sh > build.log 2>&1 

if [[ $? == 1 ]]; then
    echo "build failed"
    cat build.log | grep -iE "failed|error"
    exit 1
fi
