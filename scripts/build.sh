#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname $(readlink -f $0))

cd ${BASEDIR}/..

pushd python/paddle_edl/protos/
bash generate.sh
popd

#build master go
mkdir -p build/cmd/master/
go build   -o build/master/master cmd/master/master.go

#test all go test
go test --cover ./...

#build python
build_dir=build
mkdir -p  ${build_dir}
pushd ${build_dir}
cmake ..
make clean && make -j
ctest -V -R
popd

echo "complete!"
