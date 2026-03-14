# Phase 4 — Action Executors

## Goal

Replace the placeholder executors in `simulator.py` with real physics-based implementations that use the controllers from Phase 3. Each executor handles one `action` type from the workflow. The step dispatcher routes by equipment type — any scenario works automatically.

## Tasks

### 4.1 Manipulator executor (`backend/app/services/simulator.py`)

**`_scripted_manipulation(model, data, step, target_positions, catalog) → StepResult`**
- Resolve target name → 3D coordinates
- Find end-effector site via `find_ee_site(model, step.equipment_id)`
- Create `IKEngine` and `GraspManager` for this manipulator

**Action dispatch:**
- **`pick`**: IK to target position → `GraspManager.attach(nearest_object)` → IK to lift height (target + z_offset)
- **`place`**: IK to target position → `GraspManager.detach()` → step physics (object falls via gravity)
- **`move`**: IK to target position (no grasp change — moves with or without held object)

**Collision tracking:**
- Count `data.ncon` contacts per physics step
- Filter: ignore self-collisions within robot body (only count robot↔environment and robot↔object)

**Error handling:**
- IK fails to converge → `StepResult(success=False, error="Cannot reach target")`
- Object not found for pick → `StepResult(success=False, error="No graspable object near target")`
- Equipment body not in scene → `StepResult(success=False, error="Equipment not found")`

### 4.2 Conveyor executor (`backend/app/services/simulator.py`)

**`_sim_conveyor(model, data, step) → StepResult`**
- Get belt speed from `step.params.get("speed", 0.1)`
- Get conveyor velocity actuator ID by name (`{equipment_id}_belt_speed`)
- Set `data.ctrl[actuator_id] = speed`
- Step physics for `step.duration_s`
- After duration: set `data.ctrl[actuator_id] = 0` (stop belt)
- Count collisions during transport

**Alternative (if no actuator — force-based):**
- Detect contact pairs between belt geom and work objects via `data.contact`
- Apply `data.xfrc_applied[object_id] = [force_x, 0, 0, 0, 0, 0]` in belt axis direction
- Force magnitude proportional to belt speed × object mass

### 4.3 Camera executor (`backend/app/services/simulator.py`)

**`_sim_camera_inspect(model, data, step, target_positions) → StepResult`**
- Current FOV check: compute angle between camera-to-target vector and camera axis → compare with `fov_deg / 2`
- Keep this logic (it works correctly)
- Add optional: `mujoco.Renderer` → capture frame → save to `simulations/{step_order}_inspect.png`
- Return `StepResult(success=visible, image_path=...)`

### 4.4 Wait executor (`backend/app/services/simulator.py`)

- Keep current implementation: step physics for `duration_s`
- Count collisions during wait (objects settling on surfaces)

### 4.5 Update step dispatcher (`backend/app/services/simulator.py`)

**`_execute_step(model, data, step, catalog, target_positions) → StepResult`**
- Resolve equipment type from catalog
- Route to appropriate executor
- Pass controllers (IK, grasp) as needed — create per-step, reuse within step
- Catch all exceptions → wrap in `StepResult(success=False, error=str(exc))`

### 4.6 Tests

- Unit test: `_scripted_manipulation` with "pick" action → object attached to gripper
- Unit test: `_scripted_manipulation` with "place" action → object detached, falls via gravity
- Unit test: `_sim_conveyor` → objects on belt move in belt direction
- Unit test: `_sim_camera_inspect` → returns success=True for in-FOV target, False for out-of-FOV
- Integration test: full 3D print farm workflow (15 steps) → all steps complete, success_rate > 0
- Integration test: pickup point workflow (no manipulator) → conveyor + camera steps work
- Integration test: electronics repair workflow → pick + move + inspect + place sequence works

## Checkpoint

```bash
pytest backend/tests/test_simulator.py -v

# Manual: run simulation for 3D print farm
curl -X POST http://localhost:8000/api/projects/{id}/simulate
# → SimResult JSON with real step durations, collision counts
# → success_rate > 0 (some steps may still fail due to positioning — that's what iteration fixes)
```

## Commit
```
feat: real physics-based action executors for all equipment types
```
