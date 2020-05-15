#!/bin/bash
set -e
ETCD_VER=v3.4.7

# choose either URL
DOWNLOAD_URL=https://paddle-edl.bj.bcebos.com/etcd-${ETCD_VER}-linux-amd64.tar.gz

rm -f /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz
rm -rf /tmp/etcd-download-test && mkdir -p /tmp/etcd-download-test

wget -q ${DOWNLOAD_URL} -O /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz
tar xzvf /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz -C /tmp/etcd-download-test --strip-components=1
rm -f /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz

/tmp/etcd-download-test/etcd --version
/tmp/etcd-download-test/etcdctl version

mv /tmp/etcd-download-test/etcd /usr/local/bin/
mv /tmp/etcd-download-test/etcdctl /usr/local/bin/

rm -rf /tmp/etcd-download-test
