# sscbirrt: the native core

The C++20 implementation of pycbirrt's planner. The contract it implements,
and every type here, is specified in [docs/native-design.md](../docs/native-design.md);
the Python package is the reference and `tests/reference/python_reference.json`
is the oracle. `sscbirrt::core` depends on the C++20 standard library only.

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug -DSSCBIRRT_SANITIZE=ON
cmake --build build/cpp --parallel
ctest --test-dir build/cpp --output-on-failure
```

## Use it from C++

Install the package, then `find_package` it. `examples/consumer/` is a complete downstream
project and `examples/consumer/smoke.sh` runs the whole sequence against a scratch prefix.

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release -DSSCBIRRT_BUILD_TESTS=OFF
cmake --install build/cpp --prefix /path/to/install
```

```cmake
find_package(sscbirrt REQUIRED)
target_link_libraries(app PRIVATE sscbirrt::core)
```

```cpp
#include <sscbirrt/sscbirrt.hpp>
```

The library has no dependencies beyond the C++20 standard library. The Python
wheel builds the same CMake project with `SSCBIRRT_BUILD_PYTHON=ON` through
scikit-build-core.

v1.5.0 slices: joint space and sets (#116), validity, motion, and the
search (#117), the Python binding (#118), install/export, consumer, and the
parity gate (#119). v1.6.0 slices: the pose region `sscbirrt::tsr` with the
sstsr conformance corpus (#127), kinematics interfaces and the lifted set
(#128), the SSIK adapter (#129), lowering and the UR5e artifact (#130).
