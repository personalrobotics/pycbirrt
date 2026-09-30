# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Benchmark the Python and native backends with the cost broken down by component (#89, #139).

Runs the natively supported cases of the reference artifact (finite planar problems, and the UR5e TSR
problem when ssik is installed) on both backends over several seeds and records medians of wall time and
of the planner's own breakdown: roots, search, smoothing, state checks, edge checks, set samples (where IK
lives for TSR sets), and set projections. Machine-dependent; the artifact records the machine so numbers
are compared only with themselves.

    uv run python tools/benchmark_native.py                  # writes tests/reference/benchmark_native.json
    uv run python tools/benchmark_native.py --seeds 2 --quiet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "tests" / "reference" / "benchmark_native.json"


def _artifact_tool():
    spec = importlib.util.spec_from_file_location("reference_artifact", ROOT / "tools" / "reference_artifact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_artifact"] = module
    spec.loader.exec_module(module)
    return module


def _median_of(dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(d.keys() for d in dicts))
    return {k: st.median(d.get(k, 0.0) for d in dicts) for k in sorted(keys)}


def run(seeds: int) -> dict[str, Any]:
    from pycbirrt.backends import native

    tool = _artifact_tool()
    rows = []
    for case in tool.cases():
        if "skipped" in case or case["config"].abort_fn is not None or case["name"] == "timeout":
            continue
        try:
            native.lower(case["problem"], case["config"])
        except native.NativeUnsupported:
            continue
        py_wall, nat_wall, py_stats, nat_stats, py_it, nat_it = [], [], [], [], [], []
        for s in range(seeds):
            t0 = time.perf_counter()
            r = case["planner"].solve(case["problem"], seed=s)
            py_wall.append(time.perf_counter() - t0)
            py_stats.append({k: float(v) for k, v in r.stats.items()})
            py_it.append(r.iterations)
            t0 = time.perf_counter()
            lowered = native.lower(case["problem"], case["config"])
            r2 = native.solve(lowered, s, None)
            nat_wall.append(time.perf_counter() - t0)
            nat_stats.append({k: float(v) for k, v in r2.stats.items()})
            nat_it.append(r2.iterations)
        rows.append(
            {
                "case": case["name"],
                "seeds": seeds,
                "python": {
                    "wall_seconds": st.median(py_wall),
                    "iterations": st.median(py_it),
                    "stats": _median_of(py_stats),
                },
                "native": {
                    "wall_seconds": st.median(nat_wall),
                    "iterations": st.median(nat_it),
                    "stats": _median_of(nat_stats),
                },
                "speedup": st.median(py_wall) / st.median(nat_wall),
            }
        )
    return {
        "artifact": "Python versus native backend timings with the native cost breakdown",
        "issue": "https://github.com/personalrobotics/pycbirrt/issues/89",
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "versions": tool.versions(),
        "cases": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if not args.quiet:
        head = "native breakdown (ms): roots / search / smooth | state / edge / sample / project"
        print(f"{'case':36s} {'python':>10s} {'native':>10s} {'speedup':>8s}  {head}")
        for row in result["cases"]:
            n = row["native"]["stats"]
            ms = lambda k: n.get(k, 0.0) * 1e3  # noqa: E731
            phases = f"{ms('seconds_roots'):.2f} / {ms('seconds_search'):.2f} / {ms('seconds_smoothing'):.2f}"
            parts = (
                f"{ms('seconds_state_checks'):.2f} / {ms('seconds_edge_checks'):.2f} / "
                f"{ms('seconds_set_samples'):.2f} / {ms('seconds_set_projections'):.2f}"
            )
            print(
                f"{row['case']:36s} {row['python']['wall_seconds'] * 1e3:8.1f}ms "
                f"{row['native']['wall_seconds'] * 1e3:8.1f}ms {row['speedup']:7.1f}x  {phases} | {parts}"
            )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
