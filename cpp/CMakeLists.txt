# sscbirrt: the native C++20 core of pycbirrt. See docs/native-design.md.
cmake_minimum_required(VERSION 3.16)
project(sscbirrt VERSION 1.7.0 LANGUAGES CXX)

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

# The SSIK adapter: the only target that sees Eigen and ssik's header-only solvers. AUTO: build it when
# ssik_cpp (ssik >= 7.0, which ships its CMake package; pass -DSSCBIRRT_SSIK_CMAKE_DIR=$(python -c
# "import ssik; print(ssik.get_cmake_dir())") or put it on CMAKE_PREFIX_PATH) and Eigen3 are found.
set(SSCBIRRT_WITH_SSIK AUTO CACHE STRING "Build sscbirrt::ssik (ON, OFF, or AUTO)")
set(SSCBIRRT_SSIK_CMAKE_DIR "" CACHE PATH "Directory holding ssik_cppConfig.cmake (ssik.get_cmake_dir())")
set(SSCBIRRT_HAS_SSIK OFF)
if(NOT SSCBIRRT_WITH_SSIK STREQUAL "OFF")
  if(SSCBIRRT_SSIK_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${SSCBIRRT_SSIK_CMAKE_DIR}")
  elseif(SSCBIRRT_BUILD_PYTHON)
    # The wheel build: ask the build interpreter, if ssik is importable there.
    find_package(Python QUIET COMPONENTS Interpreter)
    if(Python_EXECUTABLE)
      execute_process(COMMAND "${Python_EXECUTABLE}" -c "import ssik; print(ssik.get_cmake_dir(), end='')"
        OUTPUT_VARIABLE _ssik_dir RESULT_VARIABLE _ssik_rc ERROR_QUIET)
      if(_ssik_rc EQUAL 0 AND _ssik_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_ssik_dir}")
      endif()
    endif()
  endif()
  if(SSCBIRRT_WITH_SSIK STREQUAL "ON")
    find_package(ssik_cpp CONFIG REQUIRED)
  else()
    find_package(ssik_cpp CONFIG QUIET)
  endif()
  if(TARGET ssik::ssik_cpp)
    add_library(sscbirrt_ssik STATIC src/ssik/ssik_arm.cpp)
    add_library(sscbirrt::ssik ALIAS sscbirrt_ssik)
    target_link_libraries(sscbirrt_ssik PUBLIC sscbirrt_tsr PRIVATE ssik::ssik_cpp)
    target_compile_options(sscbirrt_ssik PRIVATE -Wall -Wextra)
    set_target_properties(sscbirrt_ssik PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt_ssik EXPORT_NAME ssik)
    set(SSCBIRRT_HAS_SSIK ON)
    message(STATUS "sscbirrt: SSIK adapter enabled (ssik_cpp ${ssik_cpp_VERSION}, Eigen ${Eigen3_VERSION})")
  else()
    message(STATUS "sscbirrt: SSIK adapter disabled (ssik_cpp not found); TSR problems with SSIK report unsupported")
  endif()
endif()

# The MuJoCo scene: a separate library and, for Python, a separate module, so pycbirrt._native and
# `import pycbirrt` never touch MuJoCo. AUTO: build when a mujoco package (headers + library) is found,
# through -DSSCBIRRT_MUJOCO_DIR=<site-packages/mujoco> or the build interpreter.
set(SSCBIRRT_WITH_MUJOCO AUTO CACHE STRING "Build sscbirrt::mujoco (ON, OFF, or AUTO)")
set(SSCBIRRT_MUJOCO_DIR "" CACHE PATH "The installed mujoco Python package directory (holds include/ and libmujoco)")
set(SSCBIRRT_HAS_MUJOCO OFF)
if(NOT SSCBIRRT_WITH_MUJOCO STREQUAL "OFF")
  set(_mj_dir "${SSCBIRRT_MUJOCO_DIR}")
  if(NOT _mj_dir AND SSCBIRRT_BUILD_PYTHON)
    find_package(Python QUIET COMPONENTS Interpreter)
    if(Python_EXECUTABLE)
      execute_process(COMMAND "${Python_EXECUTABLE}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__), end='')"
        OUTPUT_VARIABLE _mj_dir RESULT_VARIABLE _mj_rc ERROR_QUIET)
      if(NOT _mj_rc EQUAL 0)
        set(_mj_dir "")
      endif()
    endif()
  endif()
  set(_mj_lib "")
  if(_mj_dir AND EXISTS "${_mj_dir}/include/mujoco/mujoco.h")
    file(GLOB _mj_libs "${_mj_dir}/libmujoco.*.dylib" "${_mj_dir}/libmujoco.so.*" "${_mj_dir}/mujoco.dll")
    list(LENGTH _mj_libs _n)
    if(_n GREATER 0)
      list(GET _mj_libs 0 _mj_lib)
    endif()
  endif()
  if(_mj_lib)
    add_library(mujoco_wheel SHARED IMPORTED)
    set_target_properties(mujoco_wheel PROPERTIES IMPORTED_LOCATION "${_mj_lib}" INTERFACE_INCLUDE_DIRECTORIES "${_mj_dir}/include")
    add_library(sscbirrt_mujoco STATIC src/mujoco/scene.cpp src/mujoco/snapshot.cpp src/mujoco/validator.cpp src/mujoco/sha256.cpp)
    add_library(sscbirrt::mujoco ALIAS sscbirrt_mujoco)
    target_link_libraries(sscbirrt_mujoco PUBLIC sscbirrt_core mujoco_wheel)
    target_compile_options(sscbirrt_mujoco PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(sscbirrt_mujoco PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt_mujoco EXPORT_NAME mujoco)
    get_filename_component(_mj_libname "${_mj_lib}" NAME)
    set(SSCBIRRT_MUJOCO_LIBNAME "${_mj_libname}")
    if(APPLE)
      # The dylib's install name is a framework path inside the wheel; anything linking it must reference
      # @rpath/<libname> instead, as the mujoco wheel's own modules do.
      execute_process(COMMAND otool -D "${_mj_lib}" OUTPUT_VARIABLE _mj_otool OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REGEX REPLACE ".*\n" "" SSCBIRRT_MUJOCO_INSTALL_NAME "${_mj_otool}")
    endif()
    function(sscbirrt_link_mujoco_wheel target)
      if(APPLE)
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND install_name_tool -change "${SSCBIRRT_MUJOCO_INSTALL_NAME}" "@rpath/${SSCBIRRT_MUJOCO_LIBNAME}" "$<TARGET_FILE:${target}>"
          COMMENT "Point ${target} at @rpath/${SSCBIRRT_MUJOCO_LIBNAME}")
      endif()
    endfunction()
    set(SSCBIRRT_HAS_MUJOCO ON)
    message(STATUS "sscbirrt: MuJoCo scene enabled (${_mj_lib})")
  elseif(SSCBIRRT_WITH_MUJOCO STREQUAL "ON")
    message(FATAL_ERROR "SSCBIRRT_WITH_MUJOCO=ON but no mujoco package with include/mujoco/mujoco.h and libmujoco was found (set SSCBIRRT_MUJOCO_DIR)")
  else()
    message(STATUS "sscbirrt: MuJoCo scene disabled (no mujoco package found)")
  endif()
endif()

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
  if(SSCBIRRT_HAS_MUJOCO)
    sscbirrt_test(test_mujoco_scene)
    target_link_libraries(test_mujoco_scene PRIVATE sscbirrt::mujoco)
    set_target_properties(test_mujoco_scene PROPERTIES BUILD_RPATH "${_mj_dir}")
    sscbirrt_link_mujoco_wheel(test_mujoco_scene)
    sscbirrt_test(test_mujoco_validator)
    target_link_libraries(test_mujoco_validator PRIVATE sscbirrt::mujoco)
    set_target_properties(test_mujoco_validator PROPERTIES BUILD_RPATH "${_mj_dir}")
    sscbirrt_link_mujoco_wheel(test_mujoco_validator)
  endif()
endif()

# The Python binding is the only target that sees Python. scikit-build-core drives this file with
# SSCBIRRT_BUILD_PYTHON=ON and puts pybind11's CMake package on the prefix path.
if(SSCBIRRT_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE bindings/pycbirrt_native.cpp)
  target_link_libraries(_native PRIVATE sscbirrt::core sscbirrt::tsr)
  if(SSCBIRRT_HAS_SSIK)
    target_link_libraries(_native PRIVATE sscbirrt::ssik)
    target_compile_definitions(_native PRIVATE SSCBIRRT_HAS_SSIK=1)
  endif()
  target_compile_options(_native PRIVATE -Wall -Wextra)
  install(TARGETS _native DESTINATION pycbirrt)

  if(SSCBIRRT_HAS_MUJOCO)
    pybind11_add_module(_native_mujoco MODULE bindings/pycbirrt_native_mujoco.cpp)
    target_link_libraries(_native_mujoco PRIVATE sscbirrt::mujoco)
    target_compile_options(_native_mujoco PRIVATE -Wall -Wextra)
    target_compile_definitions(_native_mujoco PRIVATE SSCBIRRT_MUJOCO_LIBNAME="${SSCBIRRT_MUJOCO_LIBNAME}")
    # The module lives in site-packages/pycbirrt and the library in site-packages/mujoco.
    set_target_properties(_native_mujoco PROPERTIES BUILD_WITH_INSTALL_RPATH ON
      INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path/../mujoco,$ORIGIN/../mujoco>")
    sscbirrt_link_mujoco_wheel(_native_mujoco)
    install(TARGETS _native_mujoco DESTINATION pycbirrt)
  endif()
endif()
