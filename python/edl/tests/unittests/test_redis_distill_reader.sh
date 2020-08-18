#!/bin/bash
unset https_proxy http_proxy

nohup redis-server --port 3456 2>&1 &
redis_pid=$!

# wait redis start
sleep 10

nohup python -m paddle_edl.distill.redis.server_register \
  --db_endpoints 127.0.0.1:3456 \
  --service_name DistillReaderTest \
  --server 127.0.0.1:3456  >  test_redist_distill_reader.1.log 2>&1 &
register_pid=$!

nohup python -m edl.distill.redis.balance_server --db_endpoints 127.0.0.1:3456 > test_redist_distill_reader.2.log 2>&1 &
discovery_pid=$!
# wait balance start
sleep 10

export PADDLE_DISTILL_BALANCE_SERVER=127.0.0.1:7001
export PADDLE_DISTILL_SERVICE_NAME=DistillReaderTest
export PADDLE_DISTILL_MAX_TEACHER=4
python distill_reader_test.py

kill -9 $discovery_pid $register_pid $redis_pid
