# Phase 5 — Visual Simulation Runner

## Goal

Make "Launch Viewer" play the entire workflow visually in the MuJoCo viewer. The client clicks a button and watches the robot pick objects, place them on the conveyor, and the conveyor transport them — in real-time 3D.

## Tasks

### 5.1 Visual simulation function (`backend/app/services/simulator.py`)

**`run_visual_simulation(scene_path, workflow, catalog, target_positions) → SimResult`**
- Opens MuJoCo viewer via `mujoco.viewer.launch_passive(model, data)`
- Executes each workflow step using the same executors from Phase 4
- After each `mj_step()` call: `viewer.sync()` to update the 3D view
- Playback at real-time speed: `time.sleep(model.opt.timestep)` between steps
- Viewer stays open after workflow completes (user can orbit, zoom, inspect)
- Returns same `SimResult` as headless simulation

### 5.2 Refactor executors for viewer support (`backend/app/services/simulator.py`)

**Add `viewer` parameter to executors:**
- `_scripted_manipulation(model, data, step, ..., viewer=None)`
- `_sim_conveyor(model, data, step, viewer=None)`
- `_sim_wait(model, data, duration, viewer=None)`
- When `viewer` is not None: call `viewer.sync()` after each `mj_step()`
- When `viewer` is None: headless mode (current behavior, no sync overhead)

**Extract `_step_physics(model, data, duration, viewer=None) → int`**
- Shared physics stepping loop used by all executors
- Counts collisions, syncs viewer if provided
- Returns collision count

### 5.3 Real-time pacing (`backend/app/services/simulator.py`)

**`_realtime_step(model, data, viewer) → None`**
- Single physics step + viewer sync + sleep to match real-time
- `dt = model.opt.timestep` (0.002s at 500Hz)
- Skip sleep if simulation is behind real-time (catch-up mode)

### 5.4 API endpoint update (`backend/app/api/simulate.py`)

**`POST /api/projects/{project_id}/view`**
- Load latest scene + recommendation + catalog
- Call `run_visual_simulation()` in a daemon thread (non-blocking for the API)
- Return immediately: `{"status": "viewer_launched"}`

### 5.5 Tests

- Unit test: `_step_physics` with viewer=None → returns collision count (headless works)
- Unit test: `_realtime_step` mock → verifies `viewer.sync()` is called
- Integration test: `run_visual_simulation` with a simple scene (table + box) → SimResult returned, no crash
- Integration test: `run_simulation` (headless) produces same SimResult as before Phase 5 (no regression)

## Checkpoint

```bash
pytest backend/tests/test_simulator.py -v

# Manual: launch viewer for 3D print farm project
curl -X POST http://localhost:8000/api/projects/{id}/view
# → MuJoCo window opens
# → Watch: robot arm moves to shelf, picks object, swings to conveyor, places object
# → Conveyor belt moves object toward output
# → Camera inspection step (brief pause at camera position)
# → Cycle repeats for remaining workflow steps
```

## Commit
```
feat: visual simulation runner — real-time workflow playback in MuJoCo viewer
```
