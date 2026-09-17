# SPDX-License-Identifier: MIT
import os

import numpy as np
import pytest

ROBOT = os.environ.get("GEODUDE_TEST_ROBOT", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "gafro-examples", "assets", "robots",
    "geodude", "geodude.xml")))
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


@requires_robot
def test_gafro_model_normalize_pose_returns_motor():
    from gafro import Motor

    from pycbirrt.backends.gafro import GafroRobotModel
    # A real single-arm model; normalize_pose must coerce a 4x4 or a Motor to Motor.
    model = GafroRobotModel.from_file(ROBOT)
    assert isinstance(model.normalize_pose(np.eye(4)), Motor)
    assert isinstance(model.normalize_pose(Motor()), Motor)
