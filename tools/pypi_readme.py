# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Pin the README's relative images and links to a git ref, for the copy that ships to PyPI.

The README in the repository uses relative paths (``docs/images/pick_yellow_seed0.gif``,
``docs/design.md``), so images and links work on GitHub, on every branch and pull request, and in a local
Markdown preview. PyPI renders the package description with no repository to resolve them against, so the
release workflow rewrites them, in the build's working copy only, to URLs pinned to the release tag: images to
``raw.githubusercontent.com``, other links to the file on GitHub. Pinned to a tag, they never change or vanish.

    python tools/pypi_readme.py --ref v3.1.0            # rewrite README.md in place
    python tools/pypi_readme.py --ref v3.1.0 --check    # print the rewritten text; change nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = "personalrobotics/sscbirrt"
IMAGE_SUFFIXES = (".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp")
_ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)  # a scheme, an anchor, or a site-root path


def pin(text: str, ref: str) -> str:
    """Rewrite every relative ``src="..."`` and Markdown link target to a URL pinned to ``ref``."""

    def url(path: str) -> str:
        if _ABSOLUTE.match(path):
            return path
        clean = path.removeprefix("./")
        if clean.lower().split("#")[0].split("?")[0].endswith(IMAGE_SUFFIXES):
            return f"https://raw.githubusercontent.com/{REPO}/{ref}/{clean}"
        return f"https://github.com/{REPO}/blob/{ref}/{clean}"

    text = re.sub(r'src="([^"]+)"', lambda m: f'src="{url(m.group(1))}"', text)
    return re.sub(r"(!?\[[^\]]*\])\(([^)\s]+)\)", lambda m: f"{m.group(1)}({url(m.group(2))})", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="the tag (or commit) the URLs are pinned to")
    parser.add_argument("--readme", type=Path, default=Path(__file__).resolve().parent.parent / "README.md")
    parser.add_argument("--check", action="store_true", help="print the rewritten README instead of writing it")
    args = parser.parse_args()
    pinned = pin(args.readme.read_text(), args.ref)
    if args.check:
        sys.stdout.write(pinned)
    else:
        args.readme.write_text(pinned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
