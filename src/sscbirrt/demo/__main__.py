# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""``sscbirrt-demo``: run the scenarios, print a report line per result, and write one MP4 per scenario."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from sscbirrt.demo import headless_gl_default

INSTALL_HINT = 'pip install "sscbirrt[demo]"'


def _missing_dependencies(video: bool) -> list[str]:
    needed = ["mujoco", "ssik"] + (["imageio", "imageio_ffmpeg", "PIL"] if video else [])
    missing = []
    for module in needed:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sscbirrt-demo", description="Rendered MuJoCo demos of the sscbirrt planner on a UR5e."
    )
    parser.add_argument("scenarios", nargs="*", help="scenarios to run (default: all; see --list)")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--seed", type=int, default=0, help="planner seed (default: 0)")
    parser.add_argument("--out", type=Path, default=Path("sscbirrt-demo"), help="output directory for the MP4s")
    parser.add_argument("--no-video", action="store_true", help="plan and report only; do not render")
    parser.add_argument("--max-frames", type=int, help="cap the frames per video (subsampled evenly)")
    args = parser.parse_args(argv)

    headless_gl_default()  # before anything imports mujoco
    missing = _missing_dependencies(video=not args.no_video)
    if missing:
        print(f"sscbirrt-demo needs {', '.join(missing)}: {INSTALL_HINT}", file=sys.stderr)
        return 2

    from sscbirrt.demo.scenarios import all_scenarios

    scenarios = all_scenarios()
    if args.list:
        for s in scenarios.values():
            print(f"{s.name:10s} {s.claim}")
        return 0
    unknown = [name for name in args.scenarios if name not in scenarios]
    if unknown:
        parser.error(f"unknown scenario(s) {', '.join(unknown)}; choose from {', '.join(scenarios)}")
    chosen = [scenarios[name] for name in args.scenarios] or list(scenarios.values())

    if not args.no_video:
        try:
            args.out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"cannot create the output directory {args.out}: {e}", file=sys.stderr)
            return 2

    try:
        from sscbirrt.demo.scene import menagerie_path

        menagerie_path()
    except ImportError as e:
        print(e, file=sys.stderr)
        return 2

    failures = 0
    for scenario in chosen:
        print(f"\n== {scenario.name}: {scenario.claim}")
        outcome = scenario.run(args.seed)
        for line in outcome.report:
            print(f"   {line}")
        if not outcome.ok:
            failures += 1
            continue
        if args.no_video:
            continue
        from sscbirrt.demo.render import RenderUnavailable, render_video

        target = args.out / f"{scenario.name}.mp4"
        try:
            frames = render_video(
                outcome.model, outcome.data, outcome.clips, target, camera=outcome.camera, max_frames=args.max_frames
            )
        except RenderUnavailable as e:
            print(f"\n{e}", file=sys.stderr)
            return 3
        print(f"   video: {target} ({frames} frames)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
