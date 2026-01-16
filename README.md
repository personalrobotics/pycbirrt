# pycbirrt

CBiRRT (Constrained Bi-directional Rapidly-exploring Random Tree) motion planner with TSR (Task Space Region) constraints.

## Installation

```bash
pip install -e ".[all]"
```

Or install with specific backends:

```bash
pip install -e ".[mujoco,eaik]"
```

## Dependencies

### MuJoCo Menagerie (required for examples)

The examples use robot models from MuJoCo Menagerie. Clone it separately:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Set the environment variable to point to your clone:

```bash
export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie
```

### TSR Library

The [TSR library](https://github.com/personalrobotics/tsr) is installed automatically as a dependency.

### EAIK

[EAIK](https://github.com/OstermD/EAIK) provides analytical inverse kinematics. Install with:

```bash
pip install eaik
```

## Quick Start

```python
from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.mujoco import MuJoCoRobotModel, MuJoCoCollisionChecker
from pycbirrt.backends.eaik import EAIKSolver
from tsr import TSR
import numpy as np

# Load your MuJoCo model
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Create backend components
robot = MuJoCoRobotModel(model, data, ee_site="ee_site")
collision = MuJoCoCollisionChecker(model, data)
ik = EAIKSolver(urdf_path="robot.urdf")

# Create planner
planner = CBiRRT(
    robot=robot,
    ik_solver=ik,
    collision_checker=collision,
)

# Define goal as TSR
goal_tsr = TSR(
    T0_w=np.eye(4),      # TSR frame at world origin
    Tw_e=np.eye(4),      # End-effector alignment
    Bw=np.array([        # Bounds: allow some position tolerance
        [-0.01, 0.01],   # x
        [-0.01, 0.01],   # y
        [-0.01, 0.01],   # z
        [0, 0],          # roll
        [0, 0],          # pitch
        [-np.pi, np.pi], # yaw (free rotation)
    ])
)

# Plan
start_config = np.zeros(6)
path = planner.plan(start_config, goal_tsrs=[goal_tsr])

if path is not None:
    print(f"Found path with {len(path)} waypoints")
```

## Examples

See the `examples/` directory:

- `ur5e_mujoco.py` - UR5e + Robotiq gripper pick planning

## References

- Berenson, D., Srinivasa, S., Ferguson, D., & Kuffner, J. (2009). Manipulation planning on constraint manifolds. ICRA.
- Berenson, D., Srinivasa, S., & Kuffner, J. (2011). Task Space Regions: A framework for pose-constrained manipulation planning. IJRR.
