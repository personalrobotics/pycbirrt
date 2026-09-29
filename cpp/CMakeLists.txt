# sscbirrt: the native C++20 core of pycbirrt. See docs/native-design.md.
cmake_minimum_required(VERSION 3.16)
project(sscbirrt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SSCBIRRT_BUILD_TESTS "Build the ctest targets" ON)
option(SSCBIRRT_SANITIZE "Build core and tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# The core depends on the C++20 standard library and nothing else (design: dependency boundary).
add_library(sscbirrt_core STATIC
  src/space.cpp
  src/sets.cpp)
add_library(sscbirrt::core ALIAS sscbirrt_core)
target_include_directories(sscbirrt_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(sscbirrt_core PUBLIC cxx_std_20)
target_compile_options(sscbirrt_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(sscbirrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt)

if(SSCBIRRT_SANITIZE)
  set(SSCBIRRT_SANITIZE_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  target_compile_options(sscbirrt_core PUBLIC ${SSCBIRRT_SANITIZE_FLAGS})
  target_link_options(sscbirrt_core PUBLIC ${SSCBIRRT_SANITIZE_FLAGS})
endif()

if(SSCBIRRT_BUILD_TESTS)
  enable_testing()
  function(sscbirrt_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE sscbirrt::core)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  sscbirrt_test(test_space)
  sscbirrt_test(test_sets)
endif()
