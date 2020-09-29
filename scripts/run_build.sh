#!/bin/bash
set -e
unset GREP_OPTIONS
BASEDIR=$(dirname "$(readlink -f "${0}")")


echo "base_dir:${BASEDIR}"
cd "${BASEDIR}"

# 2.7 is deprecated
# ./build.sh 2.7

function abort(){
    echo "Your change doesn't follow Edl's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

function check_style() {
    trap 'abort' 0

    set +e
    upstream_url='https://github.com/elasticdeeplearning/edl'
    git remote remove upstream
    git remote add upstream $upstream_url
    set -e
    git fetch upstream develop

    pre-commit install
    changed_files="$(git diff --name-only upstream/develop)"
    echo "$changed_files" | xargs pre-commit run --files

    trap : 0
}

pushd "${BASEDIR}/../"
check_style
popd


./build.sh 3.7
