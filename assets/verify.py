# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Check an installed sscbirrt-assets: exactly the manifest's files, each hash matching, both MJCFs compiling.

Run with the Python that has the wheel (and mujoco) installed, from anywhere:

    python assets/verify.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import sscbirrt_assets

manifest = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())
root = sscbirrt_assets.menagerie_path()
if "site-packages" not in root.parts:
    sys.exit(f"{root} is not an installed copy; install the wheel first")

shipped = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
missing, extra = set(manifest["files"]) - shipped, shipped - set(manifest["files"])
if missing or extra:
    sys.exit(f"file set differs from manifest: missing={sorted(missing)} extra={sorted(extra)}")
for rel, digest in manifest["files"].items():
    if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
        sys.exit(f"{rel}: SHA-256 mismatch")
if sscbirrt_assets.MENAGERIE_COMMIT != manifest["commit"]:
    sys.exit(f"MENAGERIE_COMMIT {sscbirrt_assets.MENAGERIE_COMMIT} != manifest commit {manifest['commit']}")

for xml in (sscbirrt_assets.ur5e_xml(), sscbirrt_assets.robotiq_2f85_xml()):
    model = mujoco.MjSpec.from_file(str(xml)).compile()
    print(f"{xml.name}: {model.nbody} bodies, {model.nmesh} meshes")
print(f"OK: {len(shipped)} files match manifest at menagerie {manifest['commit'][:7]}")
