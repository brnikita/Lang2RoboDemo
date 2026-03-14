# Phase 6 — Iteration Visibility

## Goal

Make the "Run Optimization" loop produce visible, demonstrable improvements. When Claude replaces equipment, the new model appears in the scene. When Claude moves equipment, the robot physically relocates. Metrics improve across iterations and the client can see why.

## Tasks

### 6.1 Scene rebuild on equipment replacement (`backend/app/services/iteration.py`)

**Update `apply_corrections(scene_path, corrections, catalog) → Path`:**
- On `replace_equipment`:
  1. Download new model via `download_equipment_model(new_equipment_id)`
  2. Remove old equipment `<body>` from XML
  3. Call `_add_manipulator_to_scene()` or `_add_conveyor_to_scene()` for the new equipment (reuse Phase 2 functions)
  4. Update `<actuator>` section (remove old, add new)
  5. Update `<equality>` weld constraints if manipulator changed
- On `add_equipment`: same flow as initial scene generation (Phase 2 functions)
- On `position_changes`: update `pos`/`euler` attributes (current approach — works)

### 6.2 Iteration-aware scene generation (`backend/app/services/scene.py`)

**`rebuild_scene_with_models(base_scene_path, corrections, model_dirs, catalog, output_path) → Path`**
- Load existing scene XML
- Apply corrections (position moves, additions, removals, replacements)
- For replacements/additions: include real MJCF models (not box geoms)
- Validate with `mujoco.MjModel.from_xml_path()` before saving
- Save as `v{n+1}.xml`

### 6.3 Iteration loop integration (`backend/app/services/iteration.py`)

**Update `run_iteration_loop()`:**
- After each `iterate_once()` → `apply_corrections()`:
  - If corrections include replacements → rebuild scene with real models
  - Run simulation on new scene → collect metrics
  - Log: iteration number, corrections applied, metrics before/after
- Convergence check: `success_rate >= 0.95 AND collision_count == 0`
- Pass `SimResult` with real physics data (not distance-check placeholders)

### 6.4 Iteration history enrichment (`backend/app/models/iteration.py`)

**Update `IterationLog`:**
- Add `metrics_before: SimMetrics` (metrics that triggered this iteration)
- Add `metrics_after: SimMetrics` (metrics after applying corrections)
- Frontend can show: before → after comparison per iteration

### 6.5 Optional: video recording per iteration (`backend/app/services/recorder.py`)

**`record_simulation_video(scene_path, workflow, catalog, target_positions, output_path) → Path`**
- Use `mujoco.Renderer` for offscreen rendering
- Render each physics step as a frame (at 30fps downsampled from 500Hz)
- Encode frames → mp4 via `imageio` or raw frame dump
- Save to `simulations/iteration_{n}_video.mp4`
- Allows client to compare iteration videos side-by-side

### 6.6 Tests

- Unit test: `apply_corrections` with `replace_equipment` → new model body present, old removed
- Unit test: `apply_corrections` with `position_changes` → body position updated
- Unit test: `IterationLog` includes `metrics_before` and `metrics_after`
- Integration test: run 2-iteration loop on 3D print farm scenario → `success_rate` improves between iterations
- Integration test: run iteration with `replace_equipment` (e.g., widow_x → franka) → scene loads with Franka mesh, not WidowX
- Scenario test: electronics repair — WidowX can't reach → Claude proposes Franka → iteration rebuilds with Franka → simulation succeeds

## Checkpoint

```bash
pytest backend/tests/test_iteration.py -v

# Manual: run optimization on a project where initial simulation has failures
curl -X POST http://localhost:8000/api/projects/{id}/iterate \
  -H "Content-Type: application/json" \
  -d '{"max_iterations": 3}'
# → IterateResponse JSON:
#   iterations_run: 2-3
#   converged: true (or improved metrics)
#   history: [{metrics_before: {success_rate: 0.6}, metrics_after: {success_rate: 0.9}}, ...]

# Manual: open v1.xml and v3.xml in MuJoCo viewer → visually different layouts
# Manual: if equipment was replaced → different robot mesh visible
```

## Commit
```
feat: iteration loop with real model replacement and visible improvements
```
