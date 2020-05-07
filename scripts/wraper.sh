#!/bin/bash 
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

bash ./scripts/build.sh > build.log 2>&1 

if [[ $? != 0 ]]; then
    echo "build failed"
    cat build.log | grep -iE "fail|error"
    exit 1
fi
