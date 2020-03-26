file(GLOB_RECURSE EDL_FILES collective/*.py demo/*.py demo/*.sh setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(
	OUTPUT ${EDL_BINARY_DIR}/.timestamp
	COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/edl ${EDL_BINARY_DIR}/python/
	COMMAND env ${py_env} ${PYTHON_EXECUTABLE} ./setup.py bdist_wheel
    DEPENDS ${EDL_FILES})
add_custom_target(edl_python ALL DEPENDS ${EDL_BINARY_DIR}/.timestamp)
add_subdirectory(edl/tests/unittests)
