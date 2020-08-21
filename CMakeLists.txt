cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(EDL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EDL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
SET(EDL_INSTALL_DIR ${CMAKE_BINARY_DIR}/output)
SET(CMAKE_INSTALL_RPATH "$ORIGIN" "${CMAKE_INSTALL_RPATH}")
project(edl)

option(WITH_TESTING     "Compile EDL with unit testing"        ON)
option(WITH_COVERAGE    "Compile EDL with code coverage"       OFF)
option(PY_VERSION       "Compile EDL with python3 support"     ${PY_VERSION})

# PY_VERSION
if(NOT PY_VERSION)
  set(PY_VERSION 2.7)
endif()

include(python)

IF(WITH_TESTING)
    ENABLE_TESTING()
ENDIF()

add_subdirectory(python)
