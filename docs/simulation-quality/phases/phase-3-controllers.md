# Phase 3 — Controllers (IK Engine + Grasp Manager)

## Goal

Build reusable, robot-agnostic controllers: a Jacobian-based IK engine that drives any articulated manipulator to a target pose, and a grasp manager that attaches/detaches objects to/from the gripper using MuJoCo weld constraints.

These controllers are **not** action executors — they are low-level primitives that Phase 4 executors compose.

## Tasks

### 3.1 IK engine (`backend/app/services/controllers.py`)

**`class IKEngine`**
- Initialized with `model: MjModel`, `data: MjData`, `site_name: str` (end-effector site)
- Resolves site ID from name at init time
- Stateless per step — safe to reuse across workflow steps

**`reach_target(target_pos: np.ndarray, max_steps: int = 500, tolerance: float = 0.01) → bool`**
- Iterative Jacobian transpose IK loop:
  1. Compute positional error: `target_pos - data.site_xpos[site_id]`
  2. If `norm(error) < tolerance` → return True (converged)
  3. Compute Jacobian: `mujoco.mj_jacSite(model, data, jacp, None, site_id)`
  4. Compute joint velocity: `dq = step_size * J^T @ error`
  5. Apply: `data.ctrl[actuator_ids] += dq` (position-controlled) or `data.qvel += dq` (velocity-controlled)
  6. `mujoco.mj_step(model, data)`
  7. Repeat until converged or `max_steps` reached
- Returns False if max_steps exceeded (unreachable)

**`_resolve_actuator_ids(model: MjModel) → np.ndarray`**
- Find actuator indices that drive the robot's joints (filter by joint names belonging to the manipulator body subtree)
- Needed because scene has multiple actuators (robot + conveyor)

**`_detect_control_mode(model: MjModel) → Literal["position", "velocity", "torque"]`**
- Inspect actuator `gainprm` and `biasprm` to determine control mode
- Menagerie models vary: Franka uses position, xArm uses position, some use velocity

### 3.2 Grasp manager (`backend/app/services/controllers.py`)

**`class GraspManager`**
- Initialized with `model: MjModel`, `data: MjData`, `gripper_site: str`
- Pre-allocates a weld equality constraint in the MJCF (added during scene generation, Phase 2)
- Tracks currently grasped object body ID (or None)

**`attach(object_body_name: str) → bool`**
- Find body ID by name
- Activate weld constraint: `model.eq_active[weld_id] = 1`
- Set weld parameters: body1=gripper parent, body2=object body
- Returns True if object found and attached

**`detach() → None`**
- Deactivate weld constraint: `model.eq_active[weld_id] = 0`
- Clear tracked object

**Scene preparation** (added to Phase 2 scene generator):
- Pre-allocate N weld equality constraints (N = number of work objects) in `<equality>` section, all initially `active="false"`
- Each weld references the gripper body and a placeholder object body

### 3.3 End-effector site resolution (`backend/app/services/controllers.py`)

**`find_ee_site(model: MjModel, equipment_id: str) → str`**
- Mapping of known Menagerie models → end-effector site names:
  - `ufactory_xarm7` → `"attachment_site"` or last link site
  - `franka_emika_panda` → `"end_effector"` or `"grip_site"`
  - etc.
- Fallback: find last body in the robot's kinematic chain → use its site
- Returns site name string for IK engine

### 3.4 Tests

- Unit test: `IKEngine` with a simple 2-link arm → converges to reachable target
- Unit test: `IKEngine` with unreachable target → returns False within max_steps
- Unit test: `GraspManager.attach()` activates weld, `detach()` deactivates
- Integration test: load xArm7 MJCF → `IKEngine` drives arm to target position within 500 steps
- Integration test: load Franka MJCF → same test (validates universality)
- Integration test: `GraspManager` attaches box to gripper → box follows arm movement after `mj_step()`

## Checkpoint

```bash
pytest backend/tests/test_controllers.py -v

# Manual: load xArm7 scene with a box, run IK to box position, attach, lift
python -c "
import mujoco, numpy as np
from backend.app.services.controllers import IKEngine, GraspManager

m = mujoco.MjModel.from_xml_path('test_scene.xml')
d = mujoco.MjData(m)
ik = IKEngine(m, d, 'ee_site')
reached = ik.reach_target(np.array([0.3, 0.0, 0.5]))
print(f'Reached: {reached}')
# Expect: True
"
```

## Commit
```
feat: IK engine and grasp manager — universal controllers for any manipulator
```
