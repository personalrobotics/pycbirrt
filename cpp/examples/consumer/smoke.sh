#!/usr/bin/env bash
# Install sscbirrt to a scratch prefix and build the consumer against it, as a downstream project would.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cpp=$(cd "$here/../.." && pwd)
work=${1:-$(mktemp -d)}
cmake -S "$cpp" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DSSCBIRRT_BUILD_TESTS=OFF > /dev/null
cmake --build "$work/build" --parallel > /dev/null
cmake --install "$work/build" --prefix "$work/prefix" > /dev/null
cmake -S "$here" -B "$work/consumer" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$work/prefix" > /dev/null
cmake --build "$work/consumer" --parallel > /dev/null
"$work/consumer/plan"
