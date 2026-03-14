# Plan: Presentation-Quality Simulation Engine

## Context

Lang2Robo is a **universal automation generator**. The user describes any business process in text, Claude selects equipment from the catalog, builds a scene, simulates it, and optimizes iteratively. This is a platform, not a single scenario.

SPEC scenarios (all must work):
- **3D print farm**: robot + conveyor + camera
- **Pickup point**: conveyor + camera, NO robot
- **Electronics repair**: robot + camera, NO conveyor
- **Dark kitchen**: robot + conveyor + camera

**Current problem**: All equipment renders as colored boxes in MuJoCo. No articulated robots, conveyors don't move objects, IK controller only checks distance. The client sees a stack of cubes instead of a working automation process.

**Goal**: Make simulation visually presentable for ANY scenario Claude generates, with visible automation processes and measurable improvement through the optimization loop.

---

## Gap Analysis: Current vs SPEC

### Module 1 — Space Capture (Point Cloud)

| SPEC | Current | Gap |
|---|---|---|
| Photos → 3D reconstruction → point cloud preview | pycolmap SfM → sparse PLY (300-1100 points) | Sparse, inverted Y-axis, not rescaled on calibration |
| User calibrates scale → correct dimensions | Mesh/MJCF rescaled, PLY not | Point cloud stays in wrong coordinates after calibration |
| Grid matches room size | Fixed 10x10 grid | Grid doesn't adapt to actual scene bounds |

### Module 3 — Scene Assembly

| SPEC | Current | Gap |
|---|---|---|
| Download MJCF from Menagerie → include in scene | Models downloaded but **not included** — box geom instead | **Critical**: need `<include>` of real MJCF |
| `add_equipment_to_scene(mjcf, model_path, pos, orientation)` | Generates `<body><geom type="box"/></body>` | Ignores `model_path` entirely |
| Work objects — dynamic bodies | `<freejoint>` + box geom | OK, but no grasp/attachment mechanism |
| Existing equipment — static bodies | Box geom, brown color | Acceptable for MVP |

### Module 4 — Simulation Executors

| Equipment Type | SPEC Behavior | Current | Gap |
|---|---|---|---|
| **manipulator** (pick/place/move) | `compute_ik_trajectory()` → `execute_trajectory()` | Distance check → success/fail | No IK, no joint movement, no grasp |
| **conveyor** (transport) | `find_joint()` → `set_conveyor_speed()` → `sim_until()` | `mj_step()` without any force | Objects don't move |
| **camera** (inspect) | `render_camera()` → `is_in_camera_fov()` | Geometric FOV check only | Rendering not implemented |
| **wait** | Physics stepping | `mj_step()` | OK |
| **learned policy** (MVP v2) | `policy.predict(obs)` → `apply_action()` | Not implemented | Phase 2 |

### Module 5 — Iteration

| SPEC | Current | Gap |
|---|---|---|
| Claude corrects positions → re-simulation shows improvement | XML edits → re-simulation | Works, but re-simulation shows no visible change because nothing moves |
| Equipment replacement → download new model → rebuild scene | Download works | New model also becomes a box geom |

---

## Features (by system layer)

Each feature is universal — works for any equipment combination Claude generates.

### Feature 5: Point Cloud Quality (Module 1)

**What**: Fix point cloud rendering — currently inverted, not rescaled, too sparse.

**Three root causes and fixes**:

#### 5a: Coordinate system — COLMAP vs Three.js

COLMAP (OpenCV): X-right, **Y-down**, Z-forward.
Three.js: X-right, **Y-up**, Z-toward-viewer.

Current code (`reconstruction.py`): `points.append(point3d.xyz)` — no transformation. Result: room hangs upside-down below the grid.

**Fix**: Transform in `_export_pointcloud()`:
```python
# COLMAP (X, Y, Z) → Three.js (X, -Y, -Z)
transformed = np.column_stack([points[:, 0], -points[:, 1], -points[:, 2]])
```

#### 5b: Point cloud not rescaled during calibration

`calibrate_scale()` scales mesh (OBJ) and MJCF, but **not the PLY**. After calibration, displayed dimensions update (1.24m × 1.15m), but point cloud stays at raw coordinates (12.4m × 11.5m).

**Fix**: Rescale PLY in `calibrate_scale()`:
```python
def _rescale_pointcloud(ply_path: Path, scale_factor: float) -> None:
    cloud = trimesh.load(ply_path)
    cloud.vertices *= scale_factor
    cloud.export(ply_path)
```

#### 5c: Sparse reconstruction

Current: COLMAP SfM → 300-1100 sparse feature points. These are structural keypoints — always few.

**Fix (MVP)**: Increase sparse quality:
- `max_num_features = 32768` (from 8192)
- `max_ratio = 0.8` (stricter matching)
- `min_num_matches = 15` (noise filtering)
- Add sequential matching for adjacent photos

Gives 3-5x more points without major refactoring.

#### 5d: Grid doesn't match scale

Grid: fixed `gridHelper args={[10, 20]}`. If point cloud is 1.2m wide, grid is 10x too large.

**Fix**: Dynamic grid — size = point cloud bounding box × 1.5.

**Files**: `reconstruction.py`, `capture.py`, `SceneViewer3D.tsx`

**Commit**: `fix: point cloud coordinate transform, calibration rescale, dynamic grid`

---

### Feature 1: Real Equipment Models in Scenes (Module 3)

**What**: Include real MJCF models from MuJoCo Menagerie instead of box geoms when assembling scenes.

**By equipment type**:

| Type | Approach | Source |
|---|---|---|
| **manipulator** | MuJoCo `<include file="..."/>` with relative path to downloaded model. Menagerie MJCF files are self-contained — MuJoCo resolves mesh paths relative to included file. | `robot_descriptions` package (57 MuJoCo-ready models) |
| **conveyor** | Generate parametric MJCF: belt surface geom + slide joint + velocity actuator. Dimensions from catalog (length_m, width_m). | Generated at build time |
| **camera** | Keep `<camera>` element + add small visual body for camera housing. | Generated at build time |
| **fixture** | Keep box geom — acceptable for tables/shelves. | Current approach |

**Key technical approach for manipulators**:
```xml
<!-- Scene wraps robot in positioning body -->
<body name="xarm7_base" pos="0.8 0.8 0.85" euler="0 0 90">
  <include file="../../../models/ufactory_xarm7/xarm7.xml"/>
</body>
```
MuJoCo `<include>` resolves all mesh/texture paths relative to the included file's directory. No asset merging needed.

**Key technical approach for conveyors**:
```xml
<body name="conveyor_500mm" pos="0.6 0.8 0.85">
  <geom name="belt_surface" type="box" size="0.25 0.075 0.005"
        friction="1 0.005 0.0001" rgba="0.3 0.3 0.3 1"/>
  <body name="belt_roller_left" pos="-0.25 0 0">
    <geom type="cylinder" size="0.02 0.075" rgba="0.5 0.5 0.5 1"/>
  </body>
  <body name="belt_roller_right" pos="0.25 0 0">
    <geom type="cylinder" size="0.02 0.075" rgba="0.5 0.5 0.5 1"/>
  </body>
</body>
```

**Files**: `scene.py`, `downloader.py`

**Commit**: `feat: include real MJCF models for all equipment types`

---

### Feature 2: Universal Action Executors (Module 4)

**What**: Implement real physics-based executors for each `action` type from the workflow. Works for any equipment combination.

#### 2a: Manipulator actions (pick / place / move)

For **any** manipulator from the catalog (xarm7, franka, ur5e, aloha, kinova, sawyer, widow_x, so_arm100):

- **IK controller**: Jacobian transpose method via `mj_jacSite()`. Universal — works with any number of joints/DOF. End-effector identified by `<site>` on gripper (all Menagerie models define end-effector sites).
- **Grasp**: MuJoCo weld equality constraint — attach object body to gripper site. Runtime on/off via `model.eq_active`.
- **pick** = IK to target → activate weld constraint → IK to lift height
- **place** = IK to drop position → deactivate weld → object falls via gravity
- **move** = IK to target (with or without grasped object)

```python
def ik_step(model, data, site_id, target_pos, step_size=0.05):
    """One IK iteration using Jacobian transpose."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    error = target_pos - data.site_xpos[site_id]
    dq = step_size * jacp.T @ error
    data.qvel[:model.nv] += dq
    mujoco.mj_step(model, data)
```

#### 2b: Conveyor actions (transport)

For **any** conveyor from the catalog (500mm, 1000mm, 2000mm):

- Belt velocity via force applied to objects in contact with belt surface
- Detect contact pairs with belt geom via `data.contact`
- Apply `xfrc_applied` to contacting objects in belt axis direction
- Duration controls how long the belt runs

#### 2c: Camera actions (inspect)

For **any** camera (overhead, microscope, barcode):

- FOV visibility check (current — works)
- Optional: `mujoco.Renderer` for actual frame capture → save image for review

#### 2d: Wait action

- Physics stepping (current — works)

**Files**: `simulator.py`, new `controllers.py`

**Commit**: `feat: universal action executors for all equipment types`

---

### Feature 3: Visual Simulation Mode (Module 4)

**What**: "Launch Viewer" opens MuJoCo viewer and **plays the entire workflow** visually — robot moves, conveyor runs, objects are picked and placed.

**How**:
- `mujoco.viewer.launch_passive()` — non-blocking viewer
- Loop: for each workflow step → execute controller → viewer renders
- Viewer sync via `viewer.sync()` on each physics step
- Real-time playback at ~60fps

**Files**: `simulator.py`, `simulate.py`

**Commit**: `feat: visual simulation runner with real-time MuJoCo viewer`

---

### Feature 4: Iteration Visibility (Module 5)

**What**: After "Run Optimization" — the iteration loop already works (Claude corrects → re-sim). With Features 1-3, each iteration will now be **visually different** — robot repositioned, equipment swapped, workflow changed, metrics improved.

**Additional work needed**:
- On `replace_equipment`: download new model → rebuild scene with real MJCF (not just box swap)
- Optional: video recording per iteration for comparison (`mujoco.Renderer` → frames → mp4)

**Files**: `iteration.py`, optional `recorder.py`

**Commit**: `feat: iteration loop with real model replacement and visual diff`

---

## Implementation Order

```
Feature 5: Point cloud quality fixes      ← Quick wins, immediate visual improvement
    ↓
Feature 1: Real models in scenes           ← Prerequisite for simulation
    ↓
Feature 2: Universal action executors      ← Makes simulation actually work
    ↓
Feature 3: Visual simulation mode          ← Client-facing demo
    ↓
Feature 4: Iteration visibility            ← Optimization demo
```

Each feature = separate commit with tests. After Feature 3, the product is demo-ready.

---

## Scenario Validation

### Scenario 1: 3D Print Farm (robot + conveyor + camera)

**Equipment**: ufactory_xarm7, conveyor_500mm, camera_overhead, shelving_unit_5tier, work_table_120x80
**Workflow**: inspect → pick × 5 → place × 5 → transport → inspect → transport → wait

| Feature | What the client sees |
|---|---|
| **F5** (point cloud) | Room correctly oriented, properly scaled, recognizable as a bedroom |
| **F1** (real models) | xArm7 with full mesh/joints visible on the table. Conveyor with belt surface and rollers. Shelving unit as box (fixture). Camera housing visible. |
| **F2** (executors) | xArm7 arm reaches to shelf, joints rotate, gripper descends to pick. Weld attaches print to gripper. Arm lifts and swings to conveyor. Weld releases — print drops onto belt. Belt surface drags prints via friction toward output. Camera FOV check passes/fails. |
| **F3** (visual mode) | Full cycle plays in viewer: arm sequentially picks 5 prints from shelves, places each on conveyor, belt moves them, camera inspects. ~52 second cycle visible. |
| **F4** (iteration) | Iteration 1: xArm can't reach shelf tier 5 → Claude moves robot closer → re-sim shows arm reaching all tiers. Iteration 2: collision with conveyor frame → Claude raises robot Z by 0.05m → clean run. Metrics: success_rate 0.6 → 0.85 → 1.0. |

### Scenario 2: Pickup Point (conveyor + camera, NO robot)

**Equipment**: conveyor_1000mm, camera_barcode
**Workflow**: inspect → transport → wait

| Feature | What the client sees |
|---|---|
| **F5** | Room point cloud shows shelves and reception table |
| **F1** | 1m conveyor with belt surface. Barcode camera housing. Parcels as colored boxes (work objects). No robot — no manipulator MJCF needed. |
| **F2** | Camera FOV check on reception area. Belt activates — parcels slide along via friction from reception to shelf zone. Wait step: physics runs, parcels settle. |
| **F3** | Parcels appear on belt → belt moves them → camera checks → parcels arrive at shelf zone. No pick/place actions — purely transport + inspect. |
| **F4** | Camera can't see reception_table (bad angle) → Claude adjusts camera position → re-sim: FOV check passes. Metrics: success_rate 0.67 → 1.0. |

**Key validation**: Scenario works **without any manipulator**. No IK, no grasp — only conveyor + camera executors fire. The executor dispatch (`_execute_step`) correctly routes by equipment type.

### Scenario 3: Electronics Repair (robot + camera, NO conveyor)

**Equipment**: widow_x (or koch_v1_1), camera_microscope
**Workflow**: pick → move → inspect → place

| Feature | What the client sees |
|---|---|
| **F5** | Workshop point cloud showing soldering station, table |
| **F1** | Small WidowX arm (full mesh, 6-DOF joints visible). Microscope camera housing. PCB boards as thin flat boxes. No conveyor in scene. |
| **F2** | WidowX IK: arm reaches intake table, picks PCB (weld). Arm moves (with PCB attached) to microscope FOV position. Camera inspect: FOV check. Arm moves to soldering station, places PCB (weld off). |
| **F3** | Full cycle: pick → hold → inspect → place. Three PCBs processed sequentially. |
| **F4** | WidowX can't reach soldering_station (reach 0.45m, need 0.55m) → Claude proposes `replace_equipment`: widow_x → franka_emika_panda. System downloads Franka MJCF → rebuilds scene → Franka arm appears instead of WidowX → re-sim succeeds. |

**Key validation**: Scenario works **without any conveyor**. No belt physics needed — only manipulator + camera executors fire. Equipment replacement downloads and includes a completely different robot MJCF.

### Scenario 4: Dark Kitchen (robot + conveyor + camera)

**Equipment**: franka_emika_panda, conveyor_500mm, camera_overhead
**Workflow**: pick → place → wait → inspect → pick → place → transport

| Feature | What the client sees |
|---|---|
| **F5** | Kitchen point cloud showing stove, counter, stations |
| **F1** | Franka Panda with full mesh (7 joints visible). Short conveyor with belt. Overhead camera. Food containers as boxes. |
| **F2** | Franka picks container from station 1, places at station 2. Wait (physics runs — container stays). Camera inspects portioning. Franka picks from station 2, places on conveyor input. Belt activates — container slides to output. |
| **F3** | Full 7-step workflow plays: pick → place → wait → inspect → pick → place → transport. Mixed manipulator + conveyor actions in sequence. |
| **F4** | Step 6 collision: arm hits table edge → Claude shifts Franka Z up by 0.1m → clean re-sim. Metrics: collision_count 12 → 0. |

**Key validation**: Scenario uses **all three equipment types** together. Manipulator actions interleave with wait, inspect, and transport. The executor dispatch handles mixed workflows correctly.

### Cross-Scenario Validation Summary

| Aspect | Print Farm | Pickup Point | Electronics | Dark Kitchen |
|---|---|---|---|---|
| Manipulator | xArm7 | NONE | WidowX → Franka | Franka |
| Conveyor | 500mm | 1000mm | NONE | 500mm |
| Camera | overhead | barcode | microscope | overhead |
| IK controller needed | YES | NO | YES | YES |
| Grasp/weld needed | YES | NO | YES | YES |
| Belt physics needed | YES | YES | NO | YES |
| Equipment replacement | NO | NO | YES | NO |
| Unique validation | 5-tier vertical picks | No robot at all | Model swap mid-iteration | Mixed action sequence |

**All 4 scenarios are covered** by the 5 features because:
- Features are by **system layer** (scene assembly, executors, viewer), not by specific mechanic
- Executor dispatch routes by `equipment_type` — automatically skips unused types
- Each scenario exercises a different subset of the universal engine

---

## Available Resources

### Robot Models (robot_descriptions 1.23.0)
57 MuJoCo-ready models. All catalog manipulators have MJCF:
- `xarm7_mj_description` → `ufactory_xarm7/xarm7.xml`
- `panda_mj_description` → `franka_emika_panda/panda.xml`
- `ur5e_mj_description` → `universal_robots_ur5e/ur5e.xml`
- `aloha_mj_description` → `aloha/scene.xml`
- `kinova_gen3_mj_description` → `kinova_gen3/gen3.xml`

### MuJoCo 3.6.0 Capabilities
- `mj_jacSite()` — Jacobian for IK (any DOF count)
- `mj_step()` — physics at 500Hz
- `eq_active` — runtime weld constraints for grasp
- `data.ctrl[]` — actuator control for conveyor velocity
- `data.xfrc_applied` — external forces for belt transport
- `data.contact` — contact pair detection for belt-object interaction
- `mujoco.viewer.launch_passive()` — non-blocking viewer
- `mujoco.Renderer` — offscreen rendering for camera/video

### Not Available (not needed for MVP)
- DISCOVERSE — room reconstruction (Module 1, separate task)
- LeRobot + SmolVLA — policy training (Module 6, MVP v2)

---

## Expected Demo Flow

1. Client uploads room photos → sees 3D point cloud (correctly oriented, properly scaled)
2. Describes automation scenario in text (any business process)
3. Claude generates plan — equipment list from catalog + workflow steps
4. "Confirm & Build Scene" → real MJCF models downloaded, scene assembled
5. **MuJoCo viewer shows**: articulated robot arm (real mesh), conveyor belt with rollers, work objects, camera housing
6. **"Run Simulation"** → robot arm moves to targets, picks objects, places on conveyor, belt transports objects, camera inspects
7. Metrics dashboard: cycle time, success rate, collisions, failed steps
8. **"Run Optimization"** → Claude corrects positions/equipment → re-simulation with visible improvements → metrics improve
9. Final result: optimized scene + metrics report + optional video recording

---

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Menagerie MJCF `<include>` path resolution on Windows | Scene fails to load | Use forward slashes; test each robot model in isolation |
| IK divergence for unreachable targets | Simulation hangs | Cap IK iterations (500 max); keep distance pre-check as early exit |
| Conveyor friction too strong/weak | Objects fly off or don't move | Tune friction parameters per conveyor size; clamp applied force |
| Different Menagerie models have different site naming | IK can't find end-effector | Build a site-name mapping per model in catalog; fallback to last link body |
| Weld constraint causes physics instability | Simulation explodes | Use `solref`/`solimp` parameters for soft weld; limit grasp force |
| Large mesh files slow down viewer | Poor demo performance | Menagerie models already optimized (~1-5MB each); total scene < 20MB |
