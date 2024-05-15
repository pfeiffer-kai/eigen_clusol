# This file is part of EIGENLUSOL.
#
# Copyright (c) 2024 Kai Pfeiffer
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.

macro(addUnitTest name)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    add_executable(${name} ${name}.cpp ${ARGN})
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.in.cpp)
    configure_file(${name}.in.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
    add_executable(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(SEND_ERROR "Cannot add a test ${name}, neither ${name}.cpp nor ${name}.in.cpp exists in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  find_package (eigenlusol REQUIRED)
  target_link_libraries(${name} eigenlusol)
  add_test(${name}Unit ${name})
  # Adding a project configuration file (for MSVC only)
endmacro(addUnitTest)

addUnitTest(test test.cpp)

