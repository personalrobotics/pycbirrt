# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""EAIKSolver is deprecated in favor of SSIK (#63); it warns on construction."""

import numpy as np
import pytest

eaik = pytest.importorskip("eaik")


def test_eaik_solver_warns_on_construction():
    from pycbirrt.backends.eaik import EAIKSolver

    lim = (np.full(6, -2 * np.pi), np.full(6, 2 * np.pi))
    with pytest.warns(DeprecationWarning, match="SSIKSolver"):
        EAIKSolver.for_ur5e(lim)
