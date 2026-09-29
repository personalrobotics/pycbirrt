# sscbirrt: the native C++20 core of pycbirrt. See docs/native-design.md.
cmake_minimum_required(VERSION 3.16)
project(sscbirrt VERSION 1.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SSCBIRRT_BUILD_TESTS "Build the ctest targets" ON)
option(SSCBIRRT_BUILD_PYTHON "Build the pycbirrt._native pybind11 module (the wheel build turns this on)" OFF)
option(SSCBIRRT_SANITIZE "Build core and tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# The core depends on the C++20 standard library and nothing else (design: dependency boundary).
include(GNUInstallDirs)
add_library(sscbirrt_core STATIC
  src/space.cpp
  src/sets.cpp
  src/motion.cpp
  src/config.cpp
  src/planner.cpp
  src/transform.cpp)
add_library(sscbirrt::core ALIAS sscbirrt_core)
target_include_directories(sscbirrt_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(sscbirrt_core PUBLIC cxx_std_20)
target_compile_options(sscbirrt_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(sscbirrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt EXPORT_NAME core)

# The pose-region runtime: sstsr's TSR in C++, standard library only (v1.6.0 addendum).
add_library(sscbirrt_tsr STATIC src/tsr/tsr.cpp src/tsr/tsr_set.cpp)
add_library(sscbirrt::tsr ALIAS sscbirrt_tsr)
target_link_libraries(sscbirrt_tsr PUBLIC sscbirrt_core)
target_compile_options(sscbirrt_tsr PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(sscbirrt_tsr PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt_tsr EXPORT_NAME tsr)

if(SSCBIRRT_SANITIZE)
  set(SSCBIRRT_SANITIZE_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  target_compile_options(sscbirrt_core PUBLIC ${SSCBIRRT_SANITIZE_FLAGS})
  target_link_options(sscbirrt_core PUBLIC ${SSCBIRRT_SANITIZE_FLAGS})
endif()

# Install and export the package so a downstream C++ project can
#   find_package(sscbirrt REQUIRED)
#   target_link_libraries(app PRIVATE sscbirrt::core)
# examples/consumer/ is that project, built against the installed package in CI. Toggle with -DSSCBIRRT_INSTALL=OFF.
option(SSCBIRRT_INSTALL "Install headers, the library, and the sscbirrt CMake package" ON)
if(SSCBIRRT_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(DIRECTORY include/sscbirrt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS sscbirrt_core sscbirrt_tsr EXPORT sscbirrtTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT sscbirrtTargets NAMESPACE sscbirrt:: FILE sscbirrtTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sscbirrt)
  configure_package_config_file(cmake/sscbirrtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sscbirrt)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sscbirrt)
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
  sscbirrt_test(test_motion)
  sscbirrt_test(test_planner)
  sscbirrt_test(test_tsr)
  target_link_libraries(test_tsr PRIVATE sscbirrt::tsr)
  sscbirrt_test(test_tsr_set)
  target_link_libraries(test_tsr_set PRIVATE sscbirrt::tsr)
endif()

# The Python binding is the only target that sees Python. scikit-build-core drives this file with
# SSCBIRRT_BUILD_PYTHON=ON and puts pybind11's CMake package on the prefix path.
if(SSCBIRRT_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE bindings/pycbirrt_native.cpp)
  target_link_libraries(_native PRIVATE sscbirrt::core sscbirrt::tsr)
  target_compile_options(_native PRIVATE -Wall -Wextra)
  install(TARGETS _native DESTINATION pycbirrt)
endif()
