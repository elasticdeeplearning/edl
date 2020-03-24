#file(GLOB_RECURSE SERVING_CLIENT_PY_FILES paddle_serving_client/*.py)
#set(PY_FILES ${SERVING_CLIENT_PY_FILES})
SET(PACKAGE_NAME "paddle-edl")
#set(SETUP_LOG_FILE "setup.py.client.log")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
	OUTPUT ${PADDLE_EDL_BINARY_DIR}/.timestamp
	COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/collective ${PADDLE_EDL_BINARY_DIR}/python/
	COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/demo ${PADDLE_EDL_BINARY_DIR}/python/
	COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel)
    #DEPENDS ${SERVING_CLIENT_CORE} sdk_configure_py_proto ${PY_FILES})

add_custom_target(paddle_python ALL DEPENDS paddle-edl ${PADDLE_EDL_BINARY_DIR}/.timestamp)
