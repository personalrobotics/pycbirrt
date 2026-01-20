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

    # Termination
    timeout: float = 30.0  # Wall-clock timeout in seconds
    max_iterations: int = 100000  # Safety limit (timeout is the primary control)
    tsr_tolerance: float = 1e-3  # Distance tolerance for TSR satisfaction (tree connection + path constraints)
    progress_tolerance: float = 1e-6  # Minimum progress required to continue growing

    # Tree growth parameters
    step_size: float = 0.1  # Maximum joint space step
    goal_bias: float = 0.1  # Probability of start tree sampling from goal TSR
    start_bias: float = 0.1  # Probability of goal tree sampling from start TSR

    # Extension behavior (None = CON, int = EXT with X steps)
    extend_steps: int | None = None  # Steps when growing toward random sample
    connect_steps: int | None = None  # Steps when growing toward other tree

    # Constraint projection
    max_projection_iters: int = 50  # Max iterations for projecting onto constraint manifold

    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 100

    # Angular joints (for proper distance calculation with wraparound)
    # If None, all joints are treated as linear
    # If provided, boolean array where True = angular joint (handles 2*pi wraparound)
    angular_joints: tuple[bool, ...] | None = None
