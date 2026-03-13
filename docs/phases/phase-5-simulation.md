# Phase 5 — Simulation

## Goal

Implement Module 4: run the assembled MJCF scene through MuJoCo, executing each workflow step with the appropriate controller (IK for manipulators, speed control for conveyors, render for cameras). Produce metrics.

## Tasks

### 5.1 Simulation runner (`backend/app/services/simulator.py`)

**`run_simulation(scene_path, workflow, catalog, target_positions, policy=None) → SimResult`**
- Loads MJCF scene into MuJoCo
- Iterates through workflow steps sequentially
- Dispatches each step to the appropriate executor based on equipment type
- Collects step results, computes aggregate metrics
- Records video frames (optional, for later playback)

### 5.2 Step dispatcher

**`execute_step(model, data, step, catalog, target_positions, policy) → StepResult`**
- Resolves equipment type from catalog
- Routes to: `scripted_manipulation`, `sim_conveyor`, `sim_camera_inspect`, or `sim_wait`
- Catches exceptions, returns StepResult with error info

### 5.3 Manipulator executor (scripted IK)

**`scripted_manipulation(model, data, step, target_positions) → StepResult`**
- Resolves target name → 3D coordinates via `target_positions` dict
- Computes IK trajectory to target using MuJoCo's built-in IK
- For "pick": move to target, close gripper
- For "place": move to target, open gripper
- For "move": move to target (no gripper change)
- Tracks collisions during execution

**`compute_ik_trajectory(model, data, target_pos) → list[np.ndarray]`**
- Uses `mujoco.mj_inverse` or Jacobian-based IK
- Generates waypoints from current joint config to target pose

### 5.4 Conveyor executor

**`sim_conveyor(model, data, step) → StepResult`**
- Sets conveyor joint velocity from `step.params["speed"]`
- Steps simulation for `step.duration_s`
- Returns success=True (conveyors don't fail in MVP)

### 5.5 Camera executor

**`sim_camera_inspect(model, data, step, target_positions) → StepResult`**
- Renders camera view using `mujoco.Renderer`
- Checks if target position is within camera FOV
- Returns success=visible, includes rendered image

### 5.6 Metrics computation

**`compute_metrics(results: list[StepResult]) → SimMetrics`**
- `cycle_time_s`: sum of all step durations
- `success_rate`: successful steps / total steps
- `collision_count`: sum of collisions across steps
- `failed_steps`: indices of failed steps

### 5.7 API endpoint (`backend/app/api/simulate.py`)

**`POST /api/projects/{project_id}/simulate`**
- Loads latest scene + recommendation
- Runs simulation
- Saves results to `data/projects/{id}/simulations/`
- Returns: `SimResult` JSON

### 5.8 Tests
- Unit test: `compute_metrics` with known step results
- Unit test: `sim_wait` returns correct duration
- Integration test: load a simple MJCF scene (table + box), run pick step → StepResult
- Integration test: full workflow with Franka → SimResult with metrics

## Checkpoint

```bash
pytest backend/tests/test_simulator.py -v

# Manual: run simulation on built scene
curl -X POST http://localhost:8000/api/projects/{id}/simulate
# → SimResult JSON with cycle_time, success_rate, collision_count
```

## Commit
```
feat: simulation module — MuJoCo scripted execution with metrics
```
