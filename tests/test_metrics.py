#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Configuration-space metrics: Euclidean and kinetic energy."""

import os

import numpy as np
import pytest

from pycbirrt.metrics import EuclideanMetric, KineticEnergyMetric, metric_for_model

ROBOT = os.environ.get(
    "METRIC_TEST_ROBOT",
    "/home/tobi/tmp/vvv/gafro-robot-descriptions/assets/robots/universal_robots/ur5e/ur5e.yaml")
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


@pytest.fixture
def system():
    from gafro import SystemSerialization

    return SystemSerialization.load(ROBOT)


class TestEuclideanMetric:
    def test_matches_the_two_norm(self):
        metric = EuclideanMetric()
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b = rng.normal(size=6), rng.normal(size=6)
            assert metric.distance(a, b) == pytest.approx(np.linalg.norm(b - a))

    def test_zero_for_identical(self):
        metric = EuclideanMetric()
        q = np.arange(6, dtype=float)
        assert metric.distance(q, q) == pytest.approx(0.0)


@requires_robot
class TestKineticEnergyMetric:
    def test_mass_matrix_is_symmetric_positive_definite(self, system):
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(1)
        for _ in range(10):
            M = metric.mass_matrix(rng.uniform(-np.pi, np.pi, system.get_dof()))
            np.testing.assert_allclose(M, M.T, atol=1e-9)
            assert np.all(np.linalg.eigvalsh(M) > 0)

    def test_distance_is_the_energy_quadratic_form(self, system):
        """The norm is sqrt(dq' M dq), with M sampled where evaluate_at says."""
        rng = np.random.default_rng(2)
        q = rng.uniform(-1, 1, system.get_dof())
        dq = rng.uniform(-1, 1, system.get_dof())

        start = KineticEnergyMetric(system, evaluate_at="start")
        assert start.norm(q, dq) == pytest.approx(np.sqrt(dq @ start.mass_matrix(q) @ dq))

        midpoint = KineticEnergyMetric(system, evaluate_at="midpoint")
        M_mid = midpoint.mass_matrix(q + 0.5 * dq)
        assert midpoint.norm(q, dq) == pytest.approx(np.sqrt(dq @ M_mid @ dq))

    def test_zero_for_identical(self, system):
        metric = KineticEnergyMetric(system)
        q = np.zeros(system.get_dof())
        assert metric.distance(q, q) == pytest.approx(0.0)

    def test_start_rule_is_absolutely_homogeneous(self, system):
        """Only the start rule satisfies norm(a*dq) == |a| * norm(dq) exactly.

        M is then fixed at q, so scaling dq scales the norm. The midpoint and
        average rules move the evaluation point, which is why the planner asks
        the metric to scale a step instead of dividing by its length.
        """
        metric = KineticEnergyMetric(system, evaluate_at="start")
        rng = np.random.default_rng(3)
        q = rng.uniform(-1, 1, system.get_dof())
        dq = rng.uniform(-1, 1, system.get_dof())
        for scale in (-3.0, 0.5, 2.0):
            assert metric.norm(q, scale * dq) == pytest.approx(abs(scale) * metric.norm(q, dq))

    @pytest.mark.parametrize("evaluate_at", ["start", "average", "midpoint"])
    def test_scale_to_length_hits_the_requested_length(self, system, evaluate_at):
        """The planner's step sizing must land on step_size under every rule."""
        metric = KineticEnergyMetric(system, evaluate_at=evaluate_at)
        rng = np.random.default_rng(7)
        for _ in range(25):
            q = rng.uniform(-0.9, 0.9, system.get_dof())
            direction = rng.normal(size=system.get_dof())
            step = metric.scale_to_length(q, direction, 0.2)
            assert metric.norm(q, step) == pytest.approx(0.2, rel=1e-3)

    def test_scale_to_length_keeps_the_direction(self, system):
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(8)
        q = rng.uniform(-0.9, 0.9, system.get_dof())
        direction = rng.normal(size=system.get_dof())
        step = metric.scale_to_length(q, direction, 0.15)
        cosine = (step @ direction) / (np.linalg.norm(step) * np.linalg.norm(direction))
        assert cosine == pytest.approx(1.0)

    def test_obeys_the_triangle_inequality_locally(self, system):
        """At a fixed q the metric is a norm, so it is subadditive."""
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(4)
        q = rng.uniform(-1, 1, system.get_dof())
        for _ in range(20):
            u, v = rng.normal(size=system.get_dof()), rng.normal(size=system.get_dof())
            assert metric.norm(q, u + v) <= metric.norm(q, u) + metric.norm(q, v) + 1e-9

    def test_weighs_heavy_joints_more(self, system):
        """The point of the metric: a shoulder step costs more than a wrist step."""
        metric = KineticEnergyMetric(system)
        q = np.zeros(system.get_dof())
        shoulder = np.zeros(system.get_dof())
        shoulder[0] = 0.1
        wrist = np.zeros(system.get_dof())
        wrist[-1] = 0.1
        assert metric.norm(q, shoulder) > 2.0 * metric.norm(q, wrist)

    def test_differs_from_euclidean(self, system):
        """Both are equal only if M is the identity, which it is not."""
        kinetic, euclidean = KineticEnergyMetric(system), EuclideanMetric()
        rng = np.random.default_rng(5)
        q = rng.uniform(-1, 1, system.get_dof())
        dq = rng.uniform(-1, 1, system.get_dof())
        assert kinetic.norm(q, dq) != pytest.approx(euclidean.norm(q, dq))

    def test_symmetrize_makes_distance_symmetric(self, system):
        rng = np.random.default_rng(6)
        a = rng.uniform(-1, 1, system.get_dof())
        b = rng.uniform(-1, 1, system.get_dof())

        asymmetric = KineticEnergyMetric(system, symmetrize=False)
        symmetric = KineticEnergyMetric(system, symmetrize=True)
        # M varies with q, so the one-sided form is direction-dependent...
        assert asymmetric.distance(a, b) != pytest.approx(asymmetric.distance(b, a))
        # ...while the averaged one is not.
        assert symmetric.distance(a, b) == pytest.approx(symmetric.distance(b, a))

    def test_cache_returns_equal_matrices(self, system):
        metric = KineticEnergyMetric(system)
        q = np.full(system.get_dof(), 0.3)
        np.testing.assert_allclose(metric.mass_matrix(q), metric.mass_matrix(q))

    def test_requires_indices_with_a_widener(self, system):
        with pytest.raises(ValueError, match="joint_indices is required"):
            KineticEnergyMetric(system, to_system_configuration=lambda q: q)


@requires_robot
class TestMetricForModel:
    def test_builds_for_a_single_arm_model(self, system):
        from pycbirrt.backends.gafro import GafroRobotModel

        model = GafroRobotModel(system)
        metric = metric_for_model(model)
        M = metric.mass_matrix(np.zeros(model.dof))
        assert M.shape == (model.dof, model.dof)
        assert np.all(np.linalg.eigvalsh(M) > 0)

    def test_rejects_a_model_without_a_system(self):
        class Bare:
            pass

        with pytest.raises(ValueError, match="no .system"):
            metric_for_model(Bare())


@requires_robot
class TestPlannerIntegration:
    """The planner must accept either metric and still produce valid paths."""

    @staticmethod
    def _plan(metric, system):
        from tsr import TSR

        from pycbirrt import CBiRRT, CBiRRTConfig
        from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel

        class NoCollision:
            def is_valid(self, q):
                return True

        model = GafroRobotModel(system)
        start = np.full(model.dof, 0.2)
        goal_pose = model.forward_kinematics(start + 0.3).to_transformation_matrix()
        tsr = TSR(T0_w=goal_pose, Tw_e=np.eye(4),
                  Bw=np.array([[-0.05, 0.05]] * 3 + [[-0.03, 0.03]] * 3))
        solver = GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=200,
                               tolerance=1e-5, base_configuration=model.base_configuration,
                               collision_checker=NoCollision())
        config = CBiRRTConfig(max_iterations=2000, step_size=0.2, goal_bias=0.3,
                              tsr_samples=25, angular_joints=(True,) * model.dof,
                              metric=metric)
        planner = CBiRRT(model, solver, NoCollision(), config)
        return model, planner.plan(start=start, goal_tsrs=[tsr], seed=1, return_details=True)

    def test_plans_with_the_default_euclidean_metric(self, system):
        model, result = self._plan(None, system)
        assert result.success
        assert all(wp.shape == (model.dof,) for wp in result.path)

    def test_plans_with_the_kinetic_metric(self, system):
        from pycbirrt.backends.gafro import GafroRobotModel

        metric = metric_for_model(GafroRobotModel(system))
        model, result = self._plan(metric, system)
        assert result.success
        assert all(wp.shape == (model.dof,) for wp in result.path)


@requires_robot
class TestMetricAwareSmoothing:
    """Shortcutting must be accepted on path cost, not waypoint count.

    Under a kinetic-energy metric a shortcut with fewer waypoints can cost more
    work; keeping it would undo what the metric was chosen to optimise.
    """

    def test_smoothing_never_increases_path_cost(self, system):
        from tsr import TSR

        from pycbirrt import CBiRRT, CBiRRTConfig
        from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel

        class NoCollision:
            def is_valid(self, q):
                return True

        model = GafroRobotModel(system)
        metric = metric_for_model(model)
        start = np.full(model.dof, 0.2)
        goal_pose = model.forward_kinematics(start + 0.4).to_transformation_matrix()
        tsr = TSR(T0_w=goal_pose, Tw_e=np.eye(4),
                  Bw=np.array([[-0.05, 0.05]] * 3 + [[-0.03, 0.03]] * 3))
        solver = GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=200,
                               tolerance=1e-5, base_configuration=model.base_configuration,
                               collision_checker=NoCollision())
        config = CBiRRTConfig(max_iterations=2000, step_size=0.2, goal_bias=0.3,
                              tsr_samples=25, angular_joints=(True,) * model.dof,
                              metric=metric, smooth_path=False)
        planner = CBiRRT(model, solver, NoCollision(), config)
        result = planner.plan(start=start, goal_tsrs=[tsr], seed=1, return_details=True)
        assert result.success

        def cost(path):
            return sum(metric.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

        smoothed = planner._smooth_path(list(result.path))
        assert cost(smoothed) <= cost(result.path) + 1e-9


@requires_robot
class TestVolumeElement:
    def test_is_positive_and_matches_sqrt_det(self, system):
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(9)
        for _ in range(10):
            q = rng.uniform(-1, 1, system.get_dof())
            expected = np.sqrt(np.linalg.det(metric.mass_matrix(q)))
            assert metric.volume_element(q) == pytest.approx(expected)
            assert metric.volume_element(q) > 0.0


@requires_robot
class TestMetricSampling:
    """Samples must follow sqrt(det M) rather than the joint coordinates."""

    @staticmethod
    def _planner(system, metric, metric_sampling):
        from pycbirrt import CBiRRT, CBiRRTConfig
        from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel

        class NoCollision:
            def is_valid(self, q):
                return True

        model = GafroRobotModel(system)
        solver = GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=50,
                               tolerance=1e-4, base_configuration=model.base_configuration,
                               collision_checker=NoCollision())
        config = CBiRRTConfig(metric=metric, metric_sampling=metric_sampling,
                              angular_joints=(True,) * model.dof)
        planner = CBiRRT(model, solver, NoCollision(), config)
        planner._rng = np.random.default_rng(0)
        return model, planner

    def test_samples_stay_within_limits(self, system):
        from pycbirrt.backends.gafro import GafroRobotModel

        metric = metric_for_model(GafroRobotModel(system))
        model, planner = self._planner(system, metric, True)
        lower, upper = model.joint_limits
        for _ in range(200):
            q = planner._sample_random_config()
            assert np.all(q >= lower) and np.all(q <= upper)

    def test_shifts_density_toward_heavier_regions(self, system):
        """The whole point: weighted draws should favour higher-volume regions."""
        from pycbirrt.backends.gafro import GafroRobotModel

        metric = metric_for_model(GafroRobotModel(system))
        samples = {}
        for flag in (False, True):
            _model, planner = self._planner(system, metric, flag)
            samples[flag] = np.array(
                [metric.volume_element(planner._sample_random_config()) for _ in range(600)])
        assert samples[True].mean() > samples[False].mean()

    def test_is_off_by_default(self):
        from pycbirrt import CBiRRTConfig

        assert CBiRRTConfig().metric_sampling is False

    def test_ignored_without_a_volume_element(self, system):
        """A metric with no volume element must fall back to uniform, not crash."""
        from pycbirrt.metrics import EuclideanMetric

        _model, planner = self._planner(system, EuclideanMetric(), True)
        assert planner._sample_random_config() is not None


@requires_robot
class TestNaturalGradient:
    """The Riemannian natural gradient of the squared-distance potential.

    Kyaw & Kelly, "Geometry-Aware Sampling-Based Motion Planning on Riemannian
    Manifolds" (eq. 9): grad phi = G(q)^-1 * (Euclidean gradient).
    """

    def test_points_toward_the_target(self, system):
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.uniform(-0.9, 0.9, system.get_dof())
            target = rng.uniform(-0.9, 0.9, system.get_dof())
            direction = target - q
            gradient = metric.natural_gradient(q, target)
            cosine = (gradient @ direction) / (np.linalg.norm(gradient)
                                               * np.linalg.norm(direction))
            assert cosine > 0.5, "descent direction must head toward the target"

    def test_differs_from_the_straight_line(self, system):
        """If it matched the straight line the whole exercise would be moot."""
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(1)
        cosines = []
        for _ in range(30):
            q = rng.uniform(-0.9, 0.9, system.get_dof())
            target = rng.uniform(-0.9, 0.9, system.get_dof())
            direction = target - q
            gradient = metric.natural_gradient(q, target)
            cosines.append((gradient @ direction)
                           / (np.linalg.norm(gradient) * np.linalg.norm(direction)))
        assert np.mean(cosines) < 0.99

    def test_beats_the_euclidean_gradient_alignment(self, system):
        """Raising the index with G^-1 is what makes the direction usable.

        The bare Euclidean gradient of the squared distance is badly misaligned
        under an anisotropic metric; the natural gradient is not.
        """
        metric = KineticEnergyMetric(system)
        rng = np.random.default_rng(2)
        natural, euclidean = [], []
        for _ in range(20):
            q = rng.uniform(-0.9, 0.9, system.get_dof())
            target = rng.uniform(-0.9, 0.9, system.get_dof())
            direction = target - q

            raw = np.zeros_like(q)
            step = 1e-5
            for i in range(len(q)):
                offset = np.zeros_like(q)
                offset[i] = step
                raw[i] = 0.5 * (metric.distance(q + offset, target) ** 2
                                - metric.distance(q - offset, target) ** 2) / (2 * step)
            euclidean.append((-raw @ direction)
                             / (np.linalg.norm(raw) * np.linalg.norm(direction)))
            gradient = metric.natural_gradient(q, target)
            natural.append((gradient @ direction)
                           / (np.linalg.norm(gradient) * np.linalg.norm(direction)))
        assert np.mean(natural) > np.mean(euclidean)

    def test_vanishes_at_the_target(self, system):
        metric = KineticEnergyMetric(system)
        q = np.full(system.get_dof(), 0.3)
        assert np.linalg.norm(metric.natural_gradient(q, q)) < 1e-6


@requires_robot
class TestGeodesicExtension:
    """Extending along the natural gradient instead of the straight line."""

    @staticmethod
    def _plan(system, metric, geodesic_extension, collision=None):
        from tsr import TSR

        from pycbirrt import CBiRRT, CBiRRTConfig
        from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel

        class NoCollision:
            def is_valid(self, q):
                return True

        collision = collision or NoCollision()
        model = GafroRobotModel(system)
        start = np.full(model.dof, 0.2)
        goal_pose = model.forward_kinematics(start + 0.35).to_transformation_matrix()
        tsr = TSR(T0_w=goal_pose, Tw_e=np.eye(4),
                  Bw=np.array([[-0.05, 0.05]] * 3 + [[-0.03, 0.03]] * 3))
        solver = GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=200,
                               tolerance=1e-5, base_configuration=model.base_configuration,
                               collision_checker=collision)
        config = CBiRRTConfig(max_iterations=2000, step_size=0.2, goal_bias=0.3,
                              tsr_samples=25, angular_joints=(True,) * model.dof,
                              metric=metric, geodesic_extension=geodesic_extension)
        planner = CBiRRT(model, solver, collision, config)
        return model, planner.plan(start=start, goal_tsrs=[tsr], seed=1,
                                   return_details=True)

    def test_is_off_by_default(self):
        from pycbirrt import CBiRRTConfig

        assert CBiRRTConfig().geodesic_extension is False

    def test_plans_a_valid_path(self, system):
        from pycbirrt.backends.gafro import GafroRobotModel

        metric = metric_for_model(GafroRobotModel(system))
        model, result = self._plan(system, metric, True)
        assert result.success
        lower, upper = model.joint_limits
        for waypoint in result.path:
            assert waypoint.shape == (model.dof,)
            assert np.all(waypoint >= lower - 1e-6) and np.all(waypoint <= upper + 1e-6)

    def test_ignored_without_a_natural_gradient(self, system):
        """A metric lacking natural_gradient must fall back, not crash."""
        from pycbirrt.metrics import EuclideanMetric

        _model, result = self._plan(system, EuclideanMetric(), True)
        assert result.success

    def test_ignored_without_a_metric(self, system):
        _model, result = self._plan(system, None, True)
        assert result.success
