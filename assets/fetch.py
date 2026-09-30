# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Fetch the menagerie files sscbirrt-assets ships, pinned by commit and checked by SHA-256.

The files are not committed; they are copied into ``src/sscbirrt_assets/menagerie/`` at build time
(``hatch_build.py`` calls :func:`fetch`). ``manifest.json`` is the contract: the upstream commit and
the SHA-256 of every file. Any mismatch aborts the build.

    python assets/fetch.py                           # download from GitHub at the pinned commit
    python assets/fetch.py --from ~/mujoco_menagerie # copy from a local clone (hashes still checked)
    python assets/fetch.py --write-manifest --from ~/mujoco_menagerie   # regenerate the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
DEST = HERE / "src" / "sscbirrt_assets" / "menagerie"
RAW_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{path}"

# What the manifest covers: each model's MJCF, its meshes, and its LICENSE. Not the preview .png,
# the scene.xml, or the README/CHANGELOG.
MODELS = {
    "universal_robots_ur5e": ["ur5e.xml", "LICENSE", "assets/*.obj"],
    "robotiq_2f85": ["2f85.xml", "LICENSE", "assets/*.stl"],
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_source(path: str, commit: str, local: Path | None) -> bytes:
    if local is not None:
        return (local / path).read_bytes()
    with urllib.request.urlopen(RAW_URL.format(commit=commit, path=path), timeout=60) as response:
        return response.read()


def fetch(local: Path | None = None) -> Path:
    """Place every manifest file under DEST, verified; skip files already present with the right hash."""
    manifest = json.loads(MANIFEST.read_text())
    commit = manifest["commit"]
    for path, digest in manifest["files"].items():
        target = DEST / path
        if target.is_file() and _sha256(target.read_bytes()) == digest:
            continue
        data = _read_source(path, commit, local)
        if _sha256(data) != digest:
            raise RuntimeError(f"{path}: SHA-256 mismatch against manifest (menagerie commit {commit})")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Nothing outside the manifest ships: remove stale files from an earlier manifest.
    wanted = {DEST / p for p in manifest["files"]}
    for existing in DEST.rglob("*"):
        if existing.is_file() and existing not in wanted:
            existing.unlink()
    return DEST


def write_manifest(local: Path) -> None:
    """Regenerate manifest.json from a local clone at the commit it is checked out at."""
    status = subprocess.run(
        ["git", "-C", str(local), "status", "--porcelain", "--", *MODELS], capture_output=True, text=True, check=True
    ).stdout
    tracked_changes = [line for line in status.splitlines() if not line.startswith("??")]
    if tracked_changes:
        raise RuntimeError(f"{local} has local changes to the shipped models:\n" + "\n".join(tracked_changes))
    commit = subprocess.run(
        ["git", "-C", str(local), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    files = {}
    for model, patterns in MODELS.items():
        for pattern in patterns:
            for f in sorted((local / model).glob(pattern)):
                rel = f.relative_to(local).as_posix()
                files[rel] = _sha256(f.read_bytes())
    MANIFEST.write_text(json.dumps({"commit": commit, "files": files}, indent=2) + "\n")
    print(f"wrote {MANIFEST} ({len(files)} files, commit {commit})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="local", type=Path, help="copy from a local menagerie clone")
    parser.add_argument("--write-manifest", action="store_true", help="regenerate manifest.json from --from")
    parser.add_argument("--clean", action="store_true", help="delete the fetched files")
    args = parser.parse_args()
    if args.clean:
        shutil.rmtree(DEST, ignore_errors=True)
        return 0
    if args.write_manifest:
        if args.local is None:
            parser.error("--write-manifest needs --from")
        write_manifest(args.local.expanduser())
        return 0
    dest = fetch(args.local.expanduser() if args.local else None)
    print(f"assets ready in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
