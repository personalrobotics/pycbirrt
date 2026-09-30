# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""sscbirrt-demo: the scenarios plan, the CLI reports and fails clearly, and rendering writes a playable MP4 (#163)."""

import os
import sys

import pytest

from sscbirrt.demo import headless_gl_default

pytest.importorskip("mujoco")
pytest.importorskip("ssik")
pytest.importorskip("sscbirrt_assets")
from sscbirrt.backends import native_mujoco  # noqa: E402

if not native_mujoco.available():
    pytest.skip(native_mujoco.unavailable_reason(), allow_module_level=True)

from sscbirrt.demo.__main__ import main  # noqa: E402

# CI sets this so a missing GL context fails the render test instead of skipping it.
REQUIRE_RENDER = os.environ.get("SSCBIRRT_REQUIRE_RENDER") == "1"


def test_list_names_every_scenario(capsys):
    assert main(["--list"]) == 0
    listed = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert listed == ["pick", "transport", "door"]


def test_pick_plans_and_reports_without_rendering(capsys):
    assert main(["pick", "--no-video", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "== pick:" in out and "reached: the " in out and "backend: native" in out


def test_transport_constraint_holds(capsys):
    """The claim: the upright carry stays within the constraint; the free carry, same endpoints, does not."""
    import numpy as np

    from sscbirrt.demo.scenarios.transport import TILT_LIMIT

    assert main(["transport", "--no-video", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    free = float(out.split("free carry:")[1].split("max tilt")[1].split("deg")[0])
    upright = float(out.split("upright carry:")[1].split("max tilt")[1].split("deg")[0])
    assert upright <= np.degrees(np.hypot(TILT_LIMIT, TILT_LIMIT)) + 0.1  # roll and pitch each at most TILT_LIMIT
    assert free > 10.0


def test_door_follows_the_arc(capsys):
    """The chain constraint holds (the door angle stays in range) and the chain's Python fallback is reported."""
    assert main(["door", "--no-video", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "opened to 60 deg" in out and "as a TSR chain: python" in out
    assert "TSR chains stay Python" in out and "the same set as one TSR: native" in out


def test_door_single_tsr_matches_the_chain():
    import numpy as np

    from sscbirrt.demo.scenarios.door import OPEN, door_chain, door_region

    for angle in (0.0, OPEN / 2, OPEN):
        assert np.allclose(door_region(angle, angle).sample(), door_chain(angle, angle).sample(), atol=1e-12)


def test_render_writes_a_playable_mp4(tmp_path, capsys):
    imageio = pytest.importorskip("imageio.v2")
    code = main(["pick", "--seed", "0", "--out", str(tmp_path), "--max-frames", "6"])
    if code == 3 and not REQUIRE_RENDER:
        pytest.skip("no OpenGL context for offscreen rendering here")
    assert code == 0, capsys.readouterr().err
    video = tmp_path / "pick.mp4"
    reader = imageio.get_reader(str(video), format="FFMPEG")
    frames = [f for f in reader.iter_data()]  # not list(reader): its length hint can be inf for short clips
    assert len(frames) == 6 and frames[0].shape == (720, 1280, 3)
    assert frames[0].std() > 10  # a rendered scene, not a blank frame


def test_missing_dependency_names_the_extra(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "imageio", None)  # import imageio now raises ImportError
    assert main(["pick"]) == 2
    err = capsys.readouterr().err
    assert "imageio" in err and 'pip install "sscbirrt[demo]"' in err
    assert main(["pick", "--no-video"]) == 0  # planning alone does not need it


def test_unwritable_output_directory(tmp_path, capsys):
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    assert main(["pick", "--out", str(blocker / "videos")]) == 2
    assert "cannot create the output directory" in capsys.readouterr().err


def test_unknown_scenario_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as info:
        main(["nope"])
    assert info.value.code == 2 and "unknown scenario(s) nope" in capsys.readouterr().err


@pytest.mark.parametrize(
    "platform, env, expected",
    [
        ("linux", {}, "egl"),  # headless Linux: EGL
        ("linux", {"DISPLAY": ":0"}, None),  # a display: MuJoCo's default
        ("linux", {"MUJOCO_GL": "osmesa"}, "osmesa"),  # the caller's choice wins
        ("darwin", {}, None),  # macOS renders offscreen without help
    ],
)
def test_headless_gl_default(monkeypatch, platform, env, expected):
    for key in ("MUJOCO_GL", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "platform", platform)
    headless_gl_default()
    assert os.environ.get("MUJOCO_GL") == expected
