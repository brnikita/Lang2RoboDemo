# Plan: Presentation-Quality Simulation

## Problem

The MuJoCo viewer shows only colored boxes instead of recognizable equipment. There are no articulated robot arms, no conveyor belt physics, no actual pick-and-place movements. The current simulation is a structural placeholder — it validates reachability via distance checks but doesn't visually demonstrate the automation process.

For client presentations, we need:
1. Real robot arm models (articulated, with meshes) that visibly move
2. Conveyor belts that physically transport objects
3. Pick-and-place actions the viewer can watch happening
4. The iteration loop ("Run Optimization") producing visible improvements

## Current State Analysis

| Component | Current | Target |
|---|---|---|
| **Robot arms** | Box geom (`0.15 × 0.15 × reach/2`) | Full articulated MJCF from MuJoCo Menagerie (xarm7, franka, etc.) with joints + actuators |
| **Conveyors** | Static box geom | Belt surface with velocity actuator — objects slide along |
| **Fixtures** | Static box geom | Box geom (acceptable for tables/shelves) |
| **Work objects** | Box geom + freejoint | Same (acceptable) — small colored boxes represent parts |
| **Cameras** | Geometric FOV check | Same + optional rendered frame |
| **IK controller** | Distance check only | Jacobian-based IK driving joint actuators → arm visibly reaches target |
| **Conveyor physics** | `mj_step()` only | Velocity field or slide joint moving belt → objects ride along |
| **Pick action** | Reach check → success/fail | IK to target → weld constraint (attach object to gripper) → lift |
| **Place action** | Reach check → success/fail | IK to drop point → remove weld → object falls via gravity |
| **Iteration** | XML position edits | Same, but now visible because robots move and objects transport |

## What We Already Have (but don't use)

1. **robot_descriptions 1.23.0** — 57 MuJoCo-ready models including xarm7, franka_emika_panda, ur5e, aloha, kinova_gen3, sawyer. Each has full MJCF with joints, actuators, meshes, and collision geoms.

2. **Downloader infrastructure** (`downloader.py`) — already resolves `menagerie_id` → `robot_descriptions` MJCF paths. Tested: `xarm7_mj_description.MJCF_PATH` works.

3. **Scene builder** (`scene.py`) — has `_inline_include()`, `_has_mjcf()`, `_find_mjcf()` helpers. The infrastructure to include real models exists but is bypassed in favor of box geoms.

4. **MuJoCo 3.6.0** — supports `mj_jac()` (Jacobian), `mj_step()`, contact dynamics, weld equality constraints, velocity actuators — everything needed.

## Architecture: 4 Features

### Feature 1: Include Real Robot MJCF Models in Scenes

**Problem**: `scene.py` generates `<body><geom type="box"/></body>` for manipulators instead of including the actual MJCF.

**Solution**: For equipment with type `"manipulator"`, include the downloaded MJCF model (which has joints, actuators, meshes) instead of a box geom.

**Approach**:
- `generate_mjcf_scene()` already receives `model_dirs: dict[str, Path]`
- For manipulators: use `<include file="..."/>` or inline the model's `<body>` tree into the scene at the correct position
- MuJoCo Menagerie models are self-contained MJCF with `<default>`, `<asset>`, `<worldbody>`, `<actuator>` sections
- Strategy: use MuJoCo's `<include>` directive with `<body>` wrapping for positioning, OR parse the model XML and transplant its body/actuator/asset elements into the scene

**Key challenge**: Menagerie models are standalone scenes. To embed them, we need to:
1. Copy model assets (meshes, textures) to the scene directory
2. Merge `<asset>` definitions (prefix names to avoid collisions)
3. Wrap the robot `<body>` tree inside a positioning `<body>` with the placement `pos`/`euler`
4. Merge `<actuator>` definitions

**Alternative (simpler)**: Use MuJoCo's `<attach>` or `<include>` with a relative file path. MuJoCo resolves mesh paths relative to the included file. This avoids asset merging entirely:
```xml
<body name="xarm7_base" pos="0.8 0.8 0.85">
  <include file="../../models/ufactory_xarm7/xarm7.xml"/>
</body>
```

**Files to modify**:
- `backend/app/services/scene.py` — change manipulator generation to include real MJCF
- `backend/app/services/downloader.py` — ensure models are actually downloaded (not empty dirs)

**Checks**:
- `pytest backend/tests/test_scene.py -x`
- Manual: open generated scene in MuJoCo viewer → see articulated arm with meshes

**Commit**: `feat: include real robot MJCF models in generated scenes`

---

### Feature 2: Jacobian-Based IK Controller for Manipulators

**Problem**: `simulator.py` only checks `distance > reach * 1.2` and reports success/fail. No joints move.

**Solution**: Implement a resolved-rate IK controller using MuJoCo's `mj_jac()` and `mj_step()`.

**Approach**:
```python
def _ik_step(model, data, site_name, target_pos, step_size=0.05):
    """One IK iteration: compute Jacobian, apply joint velocity, step."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    error = target_pos - data.site_xpos[site_id]
    dq = step_size * jacp.T @ error  # Jacobian transpose method
    data.qvel[:len(dq)] = dq
    mujoco.mj_step(model, data)
```

**Pick/place with weld constraint**:
- **Pick**: IK to object position → create `mj_equality` weld between gripper site and object → IK to lift position
- **Place**: IK to drop position → remove weld → object falls via gravity
- MuJoCo supports runtime weld constraints via `model.eq_active`

**Files to modify**:
- `backend/app/services/simulator.py` — replace `_scripted_manipulation()` with IK controller
- Need to define `<site>` on the robot's end-effector in the included MJCF (or use existing sites from Menagerie models)

**Checks**:
- `pytest backend/tests/test_simulator.py -x`
- Manual: run simulation → MuJoCo viewer shows arm reaching, grasping, lifting, placing

**Commit**: `feat: Jacobian-based IK controller for manipulator pick/place`

---

### Feature 3: Conveyor Belt Physics

**Problem**: Conveyor is a static box. Objects don't move on it.

**Solution**: Model conveyor as a surface with velocity actuator.

**Approach** (two options):

**Option A — Velocity field via contact friction**: Use MuJoCo's `condim=4` contact model with `solref` parameters to simulate a moving belt surface. Apply tangential velocity by setting the belt body's velocity directly.

**Option B — Slide joint actuator (simpler)**: Model the conveyor belt as a body with a `<joint type="slide" axis="1 0 0"/>` and a velocity actuator. Place objects on top; friction carries them along. This is the standard MuJoCo conveyor approach.

**Recommended: Option B**
```xml
<body name="conveyor_500mm" pos="0.6 0.8 0.85">
  <joint name="conveyor_belt" type="slide" axis="1 0 0" range="-0.25 0.25"/>
  <geom type="box" size="0.25 0.075 0.01" friction="1 0.005 0.0001"/>
</body>
<actuator>
  <velocity name="belt_speed" joint="conveyor_belt" kv="100"/>
</actuator>
```

Actually, the simplest approach: keep the belt static, but apply a **force field** to objects on the belt surface during `_sim_conveyor()` by detecting contact pairs with the belt geom and applying `xfrc_applied` to the contacting objects in the belt's axis direction.

**Files to modify**:
- `backend/app/services/scene.py` — generate conveyor MJCF with proper contact surface
- `backend/app/services/simulator.py` — `_sim_conveyor()` applies tangential forces to objects in contact with belt

**Checks**:
- Manual: open scene in viewer → place object on belt → it slides along

**Commit**: `feat: conveyor belt physics with object transport`

---

### Feature 4: Visual Simulation Runner (Viewer Integration)

**Problem**: The MuJoCo viewer (`POST /{id}/view`) opens a scene but doesn't run the workflow. It's just a static viewer.

**Solution**: Create a visual simulation mode that runs the workflow step-by-step in the MuJoCo viewer, so the user can watch the robot pick, place, and the conveyor transport in real time.

**Approach**:
- New function `run_visual_simulation()` that opens `mujoco.viewer.launch_passive()` and runs the workflow loop inside it
- Each `_execute_step()` call drives the physics while the viewer renders at 60fps
- The viewer stays open and shows the entire workflow cycle

**Files to modify**:
- `backend/app/services/simulator.py` — add `run_visual_simulation()` mode
- `backend/app/api/simulate.py` — `POST /{id}/view` calls the visual runner

**Checks**:
- Manual: click "Launch Viewer" → watch the full automation cycle play out

**Commit**: `feat: visual simulation runner with real-time viewer`

---

## Implementation Order

```
Feature 1: Real robot models in scenes
    ↓
Feature 2: IK controller (depends on joints/actuators from Feature 1)
    ↓
Feature 3: Conveyor belt physics (independent, can parallel with 2)
    ↓
Feature 4: Visual simulation runner (needs 2+3 working first)
```

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Menagerie MJCF includes conflict with scene XML | Use MuJoCo `<include>` with relative paths; test each robot model independently |
| IK divergence for unreachable targets | Keep distance pre-check; cap IK iterations; fallback to "unreachable" result |
| Conveyor force too strong/weak | Tune force magnitude; use MuJoCo's built-in friction model |
| Large mesh files slow down viewer | Menagerie models are already optimized for MuJoCo; typical scene < 10MB |
| Robot model asset path resolution on Windows | Use forward slashes in MJCF; test on Windows specifically |

## Expected Result

After all 4 features:
1. **MuJoCo viewer shows**: Room with articulated robot arm (real mesh), conveyor belt, work table, camera, work objects
2. **"Run Simulation"** computes real IK trajectories, objects get picked and placed, conveyor transports parts
3. **"Run Optimization"** iterates: Claude adjusts positions → re-simulation shows improved robot movements
4. **Client demo**: Upload room photos → get automation plan → watch robot work in 3D → optimize → present metrics

This matches the DISCOVERSE-level visual quality shown in the SPEC, except using MuJoCo Menagerie models directly instead of photorealistic reconstruction (which requires DISCOVERSE, not yet integrated).
