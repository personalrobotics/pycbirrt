from dataclasses import dataclass


@dataclass
class CBiRRTConfig:
    """Configuration for CBiRRT planner."""

    # Tree growth parameters
    max_iterations: int = 5000
    step_size: float = 0.1  # Maximum joint space step per extension
    goal_bias: float = 0.1  # Probability of sampling from goal TSR

    # Connection parameters
    connect_step_size: float = 0.05  # Step size when connecting trees
    max_connect_attempts: int = 50  # Max steps when connecting trees

    # Convergence
    goal_tolerance: float = 1e-3  # TSR distance tolerance for goal

    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 100
