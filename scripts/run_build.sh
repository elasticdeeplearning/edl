#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname "$(readlink -f "${0}")")

echo "base_dir:${BASEDIR}"
cd "${BASEDIR}"

# 2.7 is deprecated
# ./build.sh 2.7

#function check_style() {
#    trap 'abort' 0
#    set -e
#
#    if [ -x "$(command -v gimme)" ]; then
#    	eval "$(GIMME_GO_VERSION=1.8.3 gimme)"
#    fi
#
#
#    pip install cpplint pylint pytest astroid isort
#    # set up go environment for running gometalinter
#    mkdir -p $GOPATH/src/github.com/PaddlePaddle/
#    ln -sf ${PADDLE_ROOT} $GOPATH/src/github.com/PaddlePaddle/Paddle
#
#    pre-commit install
#    clang-format --version
#
#    commit_files=on
#    for file_name in `git diff --numstat upstream/$BRANCH |awk '{print $NF}'`;do
#        if ! pre-commit run --files $file_name ; then
#            git diff
#            commit_files=off
#        fi
#    done
#
#    if [ $commit_files == 'off' ];then
#        echo "code format error"
#        exit 1
#    fi
#    trap : 0
#}
#
#upstream_url='https://github.com/elasticdeeplearning/edl'
#git remote remove upstream
#git remote add upstream $upstream_url.git
#check_style

./build.sh 3.7
