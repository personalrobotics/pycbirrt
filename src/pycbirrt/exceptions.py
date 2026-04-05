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
