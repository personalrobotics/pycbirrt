# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Regenerate the README's pick panel: one GIF per seed of the ``pick`` demo scenario.

    uv run python tools/readme_gifs.py            # writes docs/images/pick_<can>_seed<N>.gif

Each GIF is rendered without the text overlay (the README captions each one), at 1024x576 scaled to 400 wide,
12 fps, the motion at 1.5x the demo's speed, through a two-pass palette with the ffmpeg that imageio-ffmpeg
ships. Needs ``sscbirrt[demo]``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from sscbirrt.demo import headless_gl_default

headless_gl_default()

import imageio_ffmpeg  # noqa: E402

from sscbirrt.demo.render import render_video  # noqa: E402
from sscbirrt.demo.scenarios import pick  # noqa: E402

# Seeds chosen from a sweep of 0-59 for variety: every can, different sides, different routes, all planned fast.
SEEDS = [0, 16, 9, 1, 17, 34]
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
FILTER = (
    "fps=12,scale=400:-1:flags=lanczos,split[a][b];"
    "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
)


def main() -> int:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for seed in SEEDS:
        outcome = pick.run(seed)
        if not outcome.ok:
            print(f"seed {seed}: {outcome.report[-1]}", file=sys.stderr)
            return 1
        can = outcome.report[1].split("the ")[1].split(" can")[0]
        target = OUT / f"pick_{can}_seed{seed}.gif"
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "clip.mp4"
            render_video(
                outcome.model,
                outcome.data,
                outcome.clips,
                mp4,
                camera=outcome.camera,
                width=1024,
                height=576,
                joint_speed=1.8,
                overlay=False,
            )
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", FILTER, "-loop", "0", str(target)],
                check=True,
            )
        print(f"{target.relative_to(OUT.parent.parent)}  {target.stat().st_size / 1e6:.2f} MB  ({outcome.report[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
