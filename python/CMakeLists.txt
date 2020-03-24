file(GLOB_RECURSE EDL_FILES collective/*.py demo/*.py demo/*.sh setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(
	OUTPUT ${PADDLE_EDL_BINARY_DIR}/.timestamp
	COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/collective ${PADDLE_EDL_BINARY_DIR}/python/
	COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/demo ${PADDLE_EDL_BINARY_DIR}/python/
	COMMAND env ${py_env} ${PYTHON_EXECUTABLE} ./setup.py bdist_wheel
    DEPENDS ${EDL_FILES})
add_custom_target(paddle_edl ALL DEPENDS ${PADDLE_EDL_BINARY_DIR}/.timestamp)
