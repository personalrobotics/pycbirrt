# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Build hook: place and verify the menagerie files before hatchling packs them."""

from __future__ import annotations

import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FetchMenagerie(BuildHookInterface):
    PLUGIN_NAME = "fetch-menagerie"

    def initialize(self, version, build_data):
        sys.path.insert(0, str(Path(self.root)))
        try:
            from fetch import fetch
        finally:
            sys.path.pop(0)
        fetch()
