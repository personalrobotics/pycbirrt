from dataclasses import dataclass


@dataclass
class CBiRRTConfig:
    """Configuration for CBiRRT planner.

    Extension behavior (EXT vs CON):
    - CON (connect): March until blocked or target reached (extension_steps=None)
    - EXT (extend): Take at most X steps toward target (extension_steps=X)

    The planner supports 4 variants based on extend_steps and connect_steps:
    - CON-CON: Both trees march until blocked (default, like RRT-Connect)
    - EXT-EXT: Both trees take limited steps
    - EXT-CON: Extend tree takes limited steps, connect tree marches
    - CON-EXT: Extend tree marches, connect tree takes limited steps
    """

    # Tree growth parameters
    max_iterations: int = 5000
    step_size: float = 0.1  # Maximum joint space step
    goal_bias: float = 0.1  # Probability of sampling from goal TSR

    # Extension behavior (None = CON, int = EXT with X steps)
    extend_steps: int | None = None  # Steps when growing toward random sample
    connect_steps: int | None = None  # Steps when growing toward other tree

    # Termination
    timeout: float | None = None  # Wall-clock timeout in seconds (None = no timeout)
    goal_tolerance: float = 1e-3  # TSR distance tolerance for goal

    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 100
