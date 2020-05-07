function(go_binary TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(go_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  string(REPLACE "${PADDLE_GO_PATH}/" "" CMAKE_CURRENT_SOURCE_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_custom_command(OUTPUT ${TARGET_NAME}_timestamp
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} build
    -o "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}"
    "./${CMAKE_CURRENT_SOURCE_REL_DIR}/${go_binary_SRCS}"
    WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/go")
  add_custom_target(${TARGET_NAME} ALL DEPENDS go_vendor ${TARGET_NAME}_timestamp ${go_binary_DEPS})
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} DESTINATION bin)
endfunction(go_binary)


function(go_test TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs DEPS)
  cmake_parse_arguments(go_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  string(REPLACE "${PADDLE_GO_PATH}" "" CMAKE_CURRENT_SOURCE_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(${TARGET_NAME} ALL DEPENDS go_vendor ${go_test_DEPS})
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} test -race
    -c -o "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}"
    ".${CMAKE_CURRENT_SOURCE_REL_DIR}"
    WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/go")
  add_test(NAME ${TARGET_NAME}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction(go_test)

