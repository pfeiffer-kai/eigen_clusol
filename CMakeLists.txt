# This file is part of EIGENLUSOL.
#
# Copyright (c) 2024 Kai Pfeiffer
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.4)

project(EIGENLUSOL)

# set(CMAKE_BUILD_TYPE Release CACHE STRING "Release RelWithDebInfo Debug DEFAULT: Release" FORCE)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")

option(SAFEGUARD "Enable safeguarding functions" OFF)
option(COMPILE_TESTS "compile the tests" OFF)
option(TIMEMEASUREMENTS "compile the tests" OFF)

add_compile_options(
  -O3 -g -Wall -Wpedantic -fPIC -std=c++14 -Wno-terminate
  )

################################################################################
if(SPARSE)
    add_definitions(-DSPARSE)
endif()
if(SAFEGUARD)
    #a number of safety tests is performed:
    # - nullspace test
    # - dLagrange multiplier test
    add_definitions(-DSAFEGUARD)
endif()
if(TIMEMEASUREMENTS)
    # solver timings
    add_definitions(-DTIMEMEASUREMENTS)
endif()

file(GLOB SRC_Eigenlusol eigenlusol/*.cpp)
file(GLOB INC_Eigenlusol eigenlusol/*.h)

add_library(eigenlusol SHARED
    ${SRC_Eigenlusol}
  )

################################################################################

find_package (Eigen3 3.2.10 REQUIRED NO_MODULE)
target_link_libraries(eigenlusol Eigen3::Eigen)
target_link_libraries(eigenlusol clusol lapack blas)

################################################################################

set_target_properties(eigenlusol PROPERTIES SOVERSION 0)

install(
	TARGETS eigenlusol
    EXPORT "eigenlusolConfig"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
install(FILES ${INC_Eigenlusol} DESTINATION "include/eigenlusol")

export(TARGETS eigenlusol
    NAMESPACE eigenlusol::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/eigenlusolConfig.cmake"
)
install(EXPORT
    eigenlusolConfig
    DESTINATION "/usr/local/lib/cmake/eigenlusol/"
    NAMESPACE eigenlusol::
)


if(COMPILE_TESTS)
         add_subdirectory(tests)
endif()
