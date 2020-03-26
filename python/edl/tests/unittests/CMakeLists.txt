function(py_test_modules TARGET_NAME)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE)
      add_test(NAME ${TARGET_NAME}
          COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${EDL_BINARY_DIR}/python ${py_test_modules_ENVS}
          COVERAGE_FILE=${EDL_BINARY_DIR}/python-coverage.data
          ${PYTHON_EXECUTABLE} -m coverage run --branch -p ${EDL_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
      add_test(NAME ${TARGET_NAME}
          COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${EDL_BINARY_DIR}/python ${py_test_modules_ENVS}
          ${PYTHON_EXECUTABLE} ${EDL_SOURCE_DIR}/.tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()

    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 350)
endfunction()

function(bash_test_modules TARGET_NAME)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS LABELS)
    cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(timeout 120)
    if(${bash_test_modules_TIMEOUT})
        set(timeout ${bash_test_modules_TIMEOUT})
    endif()

    if(WITH_COVERAGE)
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${EDL_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${bash_test_modules_ENVS}
            WITH_COVERAGE=ON COVERAGE_FILE=${EDL_BINARY_DIR}/python-coverage.data
            bash ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${EDL_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${bash_test_modules_ENVS}
            bash ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (bash_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()

    if(bash_test_modules_LABELS)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT ${timeout} LABELS ${bash_test_modules_LABELS})
    else()
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT ${timeout})
    endif()
endfunction()

set(dist_ENVS http_proxy="" https_proxy="")

# python unit test
file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)

# bash unit test
file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.sh")
string(REPLACE ".sh" "" TEST_OPS "${TEST_OPS}")
foreach(TEST_OP ${TEST_OPS})
    bash_test_modules(${TEST_OP} MODULES "${TEST_OP}.sh")
endforeach(TEST_OP)
