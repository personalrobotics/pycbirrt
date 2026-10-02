#!/usr/bin/env bash
# Install sscbirrt to a scratch prefix and build the consumer against it, as a downstream project would.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cpp=$(cd "$here/../.." && pwd)
work=${1:-$(mktemp -d)}
# sstsr's C++ core ships in its wheel (sstsr>=3.3); SSTSR_CMAKE_DIR overrides asking the Python on PATH.
sstsr=${SSTSR_CMAKE_DIR:-$(python3 -c 'import tsr; print(tsr.get_cmake_dir())')}
cmake -S "$cpp" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DSSCBIRRT_BUILD_TESTS=OFF \
  -DSSCBIRRT_SSTSR_CMAKE_DIR="$sstsr" > /dev/null
cmake --build "$work/build" --parallel > /dev/null
cmake --install "$work/build" --prefix "$work/prefix" > /dev/null
cmake -S "$here" -B "$work/consumer" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$work/prefix" \
  -Dsstsr_cpp_DIR="$sstsr" > /dev/null
cmake --build "$work/consumer" --parallel > /dev/null
"$work/consumer/plan"
