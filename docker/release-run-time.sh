#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color
GREEN='\033[0;32m'

if [[ $# != 1 ]] ; then
    echo "must set version"
    exit 0
fi

version=$1

echo -e "${GREEN} Press 'y' to release ${RED} docker version ${version} ${NC}"
while : ; do
    read -n 1 k <&1
    if [[ $k == y ]] ; then
        break
    else
        echo "exit"
        exit 0
    fi
done

echo -e "\n${GREEN} Begin to release ${RED} edl docker ${version} ${NC}\n"

unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))
cd ${BASEDIR}

bash ./build-runtime.sh $version
