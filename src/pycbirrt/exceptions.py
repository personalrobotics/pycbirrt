# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Custom exceptions for pycbirrt planner."""


class PlanningError(Exception):
    """Base exception for planning errors."""

    pass


class AllStartConfigurationsInCollision(PlanningError):
    """Raised when all provided start configurations are in collision."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} start configuration(s) in collision"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class AllGoalConfigurationsInCollision(PlanningError):
    """Raised when all provided goal configurations are in collision."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} goal configuration(s) in collision"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class AllStartConfigurationsInvalid(PlanningError):
    """Raised when all provided start configurations are invalid (collision or constraint violation)."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} start configuration(s) invalid"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class AllGoalConfigurationsInvalid(PlanningError):
    """Raised when all provided goal configurations are invalid (collision or constraint violation)."""

    def __init__(self, n_configs: int, details: list[str] | None = None):
        self.n_configs = n_configs
        self.details = details or []
        msg = f"All {n_configs} goal configuration(s) invalid"
        if details:
            msg += f": {'; '.join(details)}"
        super().__init__(msg)


class UnsupportedCapability(TypeError):
    """Raised when a state set is asked for a capability it does not provide.

    For example: sampling a multi-child ``AnyOf`` without a mixture policy,
    or projecting onto a multi-child ``AllOf`` without a projection strategy.
    """

    pass


class MotionContractError(ValueError):
    """Raised when a MotionValidator returns a LocalMotion that violates its contract.

    For example: ``reached=True`` with no configurations on a nonzero motion,
    or a final configuration that is not the exact target. These are bugs in
    the validator, not planning failures, so they are raised rather than
    treated as an unreachable motion.
    """

    pass
