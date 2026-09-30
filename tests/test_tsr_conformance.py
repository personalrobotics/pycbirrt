# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native TSR agrees with sstsr on the conformance corpus (#87, #127)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tsr import TSR

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "tests" / "reference" / "tsr_conformance.json"
_native = pytest.importorskip("sscbirrt._native")


@pytest.fixture(scope="module")
def corpus():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("tsr_conformance", ROOT / "tools" / "tsr_conformance.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tsr_conformance"] = module
    spec.loader.exec_module(module)
    return module


def _native_tsr(region):
    return _native.TSR(region["T0_w"], region["Tw_e"], region["Bw"])


def test_corpus_is_checked_in_and_current(corpus, tool):
    assert corpus["regions"]
    assert tool.main(["--check"]) == 0


def test_construction_and_continuous_bounds_agree(corpus):
    for region in corpus["regions"]:
        t = _native_tsr(region)
        assert np.allclose(t.continuous_bounds(), region["continuous_bounds"], atol=1e-12), region["name"]
        assert t.volume() == pytest.approx(region["volume"], abs=1e-12), region["name"]


def test_every_probe_agrees(corpus):
    for region in corpus["regions"]:
        t = _native_tsr(region)
        for i, p in enumerate(region["probes"]):
            where = f"{region['name']}[{i}]"
            assert t.contains(p["T"]) == p["contains"], where
            assert t.distance(p["T"]) == pytest.approx(p["distance"], abs=1e-9), where
            d, closest = t.closest_transform(p["T"])
            assert d == pytest.approx(p["distance"], abs=1e-9), where
            assert np.allclose(closest, p["closest"], atol=1e-9), where
            assert t.contains(closest) == p["closest_contained"], where


def test_construction_rejections_match_sstsr(corpus):
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
    bad_frames = []
    T = np.eye(4)
    T[3, 3] = 2.0
    bad_frames.append(T)
    T = np.eye(4)
    T[0, 0] = 1.5
    bad_frames.append(T)
    T = np.eye(4)
    T[2, 2] = -1.0
    bad_frames.append(T)
    T = np.eye(4)
    T[0, 3] = np.nan
    bad_frames.append(T)
    for T in bad_frames:
        with pytest.raises(ValueError):
            TSR(T, np.eye(4), box)
        with pytest.raises(ValueError):
            _native.TSR(T.tolist(), np.eye(4).tolist(), box.tolist())
    bad_bounds = [box.copy(), box.copy()]
    bad_bounds[0][0] = [1.0, 0.0]
    bad_bounds[1][1, 1] = np.nan
    for Bw in bad_bounds:
        with pytest.raises(ValueError):
            TSR(np.eye(4), np.eye(4), Bw)
        with pytest.raises(ValueError):
            _native.TSR(np.eye(4).tolist(), np.eye(4).tolist(), Bw.tolist())
    outer = box.copy()
    outer[5] = [3 * np.pi / 4, -3 * np.pi / 4]
    TSR(np.eye(4), np.eye(4), outer)
    _native.TSR(np.eye(4).tolist(), np.eye(4).tolist(), outer.tolist())


def test_sampling_is_six_unit_draws_in_order_and_contained(corpus):
    for region in corpus["regions"]:
        t = _native_tsr(region)
        draws = _native.unit_draws(5, 6)
        s = t.sample_xyzrpy(5)
        cont = np.array(region["continuous_bounds"])
        expected = cont[:, 0] + (cont[:, 1] - cont[:, 0]) * np.array(draws)
        expected[3:] = ((expected[3:] + np.pi) % (2 * np.pi)) - np.pi
        assert np.allclose(s, expected, atol=1e-12), region["name"]
        py = TSR(region["T0_w"], region["Tw_e"], region["Bw"])
        for T in t.sample(seed=9, count=50):
            assert py.contains(np.array(T)), region["name"]
