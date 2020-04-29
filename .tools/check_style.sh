#!/bin/bash
function abort(){
    echo "Your change doesn't follow PaddleCloud's code style." 1>&2
    echo "Please use pre-commit to reformat your code and git push again." 1>&2
    exit 1
}

trap 'abort' 0
set -e

sudo find / -name clang-format

cd $TRAVIS_BUILD_DIR
export PATH=/usr/bin:$PATH
pre-commit install
pre-commit  --version

if ! pre-commit run -a ; then
  git diff  --exit-code
  exit 1
fi

trap : 0
