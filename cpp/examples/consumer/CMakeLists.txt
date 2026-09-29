# Standalone downstream project: consumes sscbirrt through find_package, built against the
# INSTALLED package (not the source tree) to prove the install and export work end to end.
# See smoke.sh and the "C++ consumer" CI job.
cmake_minimum_required(VERSION 3.16)
project(sscbirrt_consumer LANGUAGES CXX)

find_package(sscbirrt REQUIRED)
add_executable(plan main.cpp)
target_link_libraries(plan PRIVATE sscbirrt::core)
