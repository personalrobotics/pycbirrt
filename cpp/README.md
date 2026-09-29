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

Slices, tracked under the v1.5.0 milestone: joint space and sets (#116),
validity, motion, and the search (#117), the Python binding (#118),
install/export, consumer, and the parity gate (#119).
