from dataclasses import dataclass


@dataclass
class CBiRRTConfig:
    """Configuration for CBiRRT planner."""

    # Tree growth parameters
    max_iterations: int = 5000
    step_size: float = 0.1  # Maximum joint space step for both extend and connect
    goal_bias: float = 0.1  # Probability of sampling from goal TSR
    extension_steps: int | None = None  # None = CON (march until blocked), int = EXT (max X steps)

    # Convergence
    goal_tolerance: float = 1e-3  # TSR distance tolerance for goal

    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 100
