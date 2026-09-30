# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The README's relative images and links are pinned to the release tag in the copy that ships to PyPI."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("pypi_readme", ROOT / "tools" / "pypi_readme.py")
pypi_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pypi_readme)


def test_images_and_links_are_pinned_and_absolute_ones_kept():
    text = (
        '<img src="docs/images/a.gif" alt="a"> ![b](./docs/images/b.png) [design](docs/design.md) '
        '[anchor](#install) [site](https://example.com/x.png) <img src="https://example.com/c.gif">'
    )
    out = pypi_readme.pin(text, "v3.1.0")
    assert 'src="https://raw.githubusercontent.com/personalrobotics/sscbirrt/v3.1.0/docs/images/a.gif"' in out
    assert "(https://raw.githubusercontent.com/personalrobotics/sscbirrt/v3.1.0/docs/images/b.png)" in out
    assert "(https://github.com/personalrobotics/sscbirrt/blob/v3.1.0/docs/design.md)" in out
    assert "(#install)" in out and "(https://example.com/x.png)" in out and 'src="https://example.com/c.gif"' in out


def test_the_readme_uses_relative_paths_and_every_one_exists():
    """In the repository the README must work on any branch: nothing pinned to main, every relative target present."""
    text = (ROOT / "README.md").read_text()
    assert "personalrobotics/sscbirrt/main/" not in text and "sscbirrt/blob/main/" not in text
    targets = re.findall(r'src="([^"]+)"', text) + re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text)
    relative = [t for t in targets if not re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", t, re.I)]
    assert relative, "expected relative images and links"
    missing = [t for t in relative if not (ROOT / t.split("#")[0]).exists()]
    assert not missing, missing


def test_pinning_the_readme_leaves_nothing_relative():
    out = pypi_readme.pin((ROOT / "README.md").read_text(), "v9.9.9")
    targets = re.findall(r'src="([^"]+)"', out) + re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", out)
    assert all(re.match(r"^(?:https://|#)", t) for t in targets), [
        t for t in targets if not t.startswith(("https://", "#"))
    ]
