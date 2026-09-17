# SPDX-License-Identifier: MIT
import os

import numpy as np
import pytest

ROBOT = os.environ.get("GEODUDE_TEST_ROBOT", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "gafro-examples", "assets", "robots",
    "geodude", "geodude.xml")))
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


@pytest.fixture
def model():
    from pycbirrt.backends.gafro_bimanual import GafroBimanualModel
    return GafroBimanualModel.from_file(ROBOT)


@requires_robot
def test_both_active_recovers_configuration(model):
    from pycbirrt.backends.gafro_bimanual import GafroBimanualIKSolver
    rng = np.random.default_rng(0)
    lower, upper = model.joint_limits
    q_true = lower + (upper - lower) * rng.random(model.dof)
    target = model.forward_kinematics(q_true)  # both components present
    solver = GafroBimanualIKSolver(model, max_iterations=500, tolerance=1e-5)
    sols = solver.solve(target, q_init=q_true + 0.05 * rng.standard_normal(model.dof))
    assert sols, "IK did not converge"
    reached = model.forward_kinematics(sols[0])
    err_abs = np.linalg.norm(
        np.asarray(target.absolute.multiply(reached.absolute.inverse()).log()))
    err_rel = np.linalg.norm(
        np.asarray(target.relative.multiply(reached.relative.inverse()).log()))
    assert err_abs < 1e-3 and err_rel < 1e-3


@requires_robot
def test_relative_only_ignores_absolute(model):
    from tsr.bimanual import BimanualPose

    from pycbirrt.backends.gafro_bimanual import GafroBimanualIKSolver
    rng = np.random.default_rng(1)
    lower, upper = model.joint_limits
    q_true = lower + (upper - lower) * rng.random(model.dof)
    full = model.forward_kinematics(q_true)
    target = BimanualPose(absolute=None, relative=full.relative)  # relative only
    solver = GafroBimanualIKSolver(model, max_iterations=500, tolerance=1e-5)
    sols = solver.solve(target, q_init=q_true + 0.05 * rng.standard_normal(model.dof))
    assert sols
    reached = model.forward_kinematics(sols[0])
    err_rel = np.linalg.norm(
        np.asarray(target.relative.multiply(reached.relative.inverse()).log()))
    assert err_rel < 1e-3
