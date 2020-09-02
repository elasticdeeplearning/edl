#!/bin/bash
name=${TEST_TARGET_NAME}
TEST_TIMEOUT=${TEST_TIMEOUT}

if [[ ${name}"x" == "x" ]]; then
    echo "can't find ${name}, please set ${TEST_TARGET_NAME} first"
    exit 1
fi

if [[ ${TEST_TIMEOUT}"x" == "x" ]]; then
    echo "can't find ${TEST_TIMEOUT}, please set ${TEST_TIMEOUT} first"
    exit 1
fi

# rm flag file
rm -f ${name}_*.log

nohup etcd > ${name}_etcd.log 2>&1 &
etcd_pid=$!

# start the unit test
run_time=$(( $TEST_TIMEOUT - 10 ))
echo "run_time: ${run_time}"

timeout -s SIGKILL ${run_time} python -u ${name}.py > ${name}_run.log 2>&1
exit_code=$?

kill -9 $etcd_pid

echo "${name} faild with ${exit_code}"
if [[ $exit_code -eq 0 ]]; then
    exit 0
fi

echo "${name} log"
for log in `ls ${name}_*.log`
do
    printf "\ncat ${log}\n"
    cat -n ${log}
done

exit 1
