# SPDX-License-Identifier: MIT
import os

import numpy as np
import pytest

ROBOT = os.environ.get("GEODUDE_TEST_ROBOT", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "gafro-examples", "assets", "robots",
    "geodude", "geodude.xml")))
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


class _AllValid:
    def is_valid(self, q):
        return True


@requires_robot
def test_bimanual_plan_keeps_relative_grasp():
    from tsr import TSR
    from tsr.bimanual import BimanualTSR

    from pycbirrt.backends.gafro_bimanual import GafroBimanualIKSolver, GafroBimanualModel
    from pycbirrt.config import CBiRRTConfig
    from pycbirrt.planner import CBiRRT

    model = GafroBimanualModel.from_file(ROBOT)
    ik = GafroBimanualIKSolver(model, max_iterations=300, tolerance=1e-4)
    collision = _AllValid()

    # Start from the controlled-limit midpoint; read off the current abs/rel poses.
    lower, upper = model.joint_limits
    q_start = 0.5 * (lower + upper)
    start_pose = model.forward_kinematics(q_start)

    # Relative grasp: pin all 6 DOF to the current relative pose (zero-width box).
    relative_tsr = TSR(T0_w=start_pose.relative, Tw_e=np.eye(4), Bw=np.zeros((6, 2)))
    # Absolute goal region: allow the object to translate in a small box around
    # a shifted center; rotation pinned.
    goal_center = start_pose.absolute.multiply(__import__("gafro").Motor.exp(0, 0, 0, 0.1, 0.0, 0.0))
    abs_Bw = np.array([[0.0, 0.0]] * 3 + [[-0.05, 0.05]] * 3)
    absolute_goal = TSR(T0_w=goal_center, Tw_e=np.eye(4), Bw=abs_Bw)

    goal_tsr = BimanualTSR(absolute=absolute_goal, relative=relative_tsr)
    constraint_tsr = BimanualTSR(relative=relative_tsr)  # keep grasp along the path

    planner = CBiRRT(model, ik, collision, CBiRRTConfig(max_iterations=2000, timeout=30.0))
    result = planner.plan(
        start=q_start,
        goal_tsrs=[goal_tsr],
        constraint_tsrs=[constraint_tsr],
        seed=0,
        return_details=True,
    )

    assert result.success, f"bimanual plan failed: {result.failure_reason}"
    # Every waypoint keeps the relative grasp within tolerance.
    for q in result.path:
        pose = model.forward_kinematics(q)
        dist, _ = constraint_tsr.distance(pose)
        assert dist < 1e-2
