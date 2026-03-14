# Phase 7 — QA End-to-End Testing

## Goal

Validate the full pipeline works end-to-end with real room photos. Each step must complete without errors, produce correct artifacts, and the MuJoCo viewer must show a visually coherent automation scene.

Test photos: `C:\Users\Nikita\Documents\room\` (16 photos of a home office/bedroom).

## QA Test Cases

### QA-1: Photo Upload + Reconstruction

**Steps:**
1. Navigate to `/new`
2. Upload 10+ photos from the room directory
3. Wait for reconstruction to complete

**Expected:**
- API returns 200 with `project_id` and `dimensions`
- `data/projects/{id}/photos/` contains uploaded images
- `data/projects/{id}/reconstruction/pointcloud.ply` exists and > 5KB
- `data/projects/{id}/reconstruction/mesh.obj` exists
- `data/projects/{id}/status.json` exists with `current_phase: "upload"`

**Checkpoint:**
```bash
PROJECT_ID=<from response>
ls data/projects/$PROJECT_ID/photos/ | wc -l  # >= 10
ls -la data/projects/$PROJECT_ID/reconstruction/pointcloud.ply  # > 5KB
cat data/projects/$PROJECT_ID/status.json | python -m json.tool | grep current_phase
```

---

### QA-2: Point Cloud Rendering + Calibration

**Steps:**
1. On Calibrate page, verify point cloud renders
2. Check: point cloud is ABOVE the grid (not below)
3. Check: room shape is recognizable (walls, furniture outlines)
4. Check: grid size approximately matches point cloud extent
5. Enter calibration: Point A = (0,0,0), Point B = (1,0,0), Distance = 0.9m
6. Click "Calibrate & Analyze"

**Expected:**
- Point cloud renders above grid plane (Y-up, not Y-down)
- After calibration, dimensions update to realistic room size (~3-5m × 3-5m)
- `space_model.json` is created with zones, equipment, doors, windows
- `status.json` updated to `current_phase: "calibrate"`
- Point cloud file is rescaled (smaller than pre-calibration)

**Checkpoint:**
```bash
cat data/projects/$PROJECT_ID/space_model.json | python -m json.tool | head -20
cat data/projects/$PROJECT_ID/status.json | python -m json.tool | grep current_phase
```

---

### QA-3: Recommendation Generation

**Steps:**
1. On Plan page, enter scenario text:
   `"Small workshop with desk. Robot picks parts from intake tray, moves to camera inspection point, places on work table for processing."`
2. Click "Generate Plan"
3. Verify plan appears with equipment list, workflow steps, text plan

**Expected:**
- Recommendation JSON includes equipment from catalog (IDs validated)
- All `equipment_id` values exist in knowledge-base
- `workflow_steps` have valid action types (pick/place/move/inspect/wait)
- `target_positions` map referenced by all workflow steps
- `recommendation.json` saved to project directory
- `status.json` updated to `current_phase: "recommend"`

**Checkpoint:**
```bash
cat data/projects/$PROJECT_ID/recommendation/recommendation.json | python -m json.tool | head -30
```

---

### QA-4: Scene Build

**Steps:**
1. Click "Confirm & Build Scene"
2. Wait for scene assembly

**Expected:**
- Equipment models downloaded to `models/{id}/` (not empty)
- For manipulators: `find_mjcf_in_dir()` returns robot XML (not scene.xml)
- `scenes/v1.xml` generated with:
  - `<include>` for manipulator MJCF (not box geom)
  - Parametric conveyor body (if conveyor in plan)
  - Camera housing body
  - Work objects with `<freejoint>`
- Scene loads in MuJoCo without errors
- `model.njnt > 5` (robot joints present)
- `model.nsite > 0` (end-effector site present)
- `model.nu > 0` (actuators present)
- `status.json` updated to `current_phase: "build-scene"`

**Checkpoint:**
```bash
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('data/projects/$PROJECT_ID/scenes/v1.xml')
print(f'Bodies: {m.nbody}, Joints: {m.njnt}, Sites: {m.nsite}, Actuators: {m.nu}')
assert m.njnt > 5, 'No robot joints'
assert m.nsite > 0, 'No EE site'
assert m.nu > 0, 'No actuators'
print('Scene validation PASSED')
"
```

---

### QA-5: Simulation Run

**Steps:**
1. Click "Run Simulation"
2. Wait for simulation to complete

**Expected:**
- No 500 Internal Server Error
- `SimResult` returned with:
  - `steps`: list of StepResult per workflow step
  - `metrics.cycle_time_s > 0`
  - `metrics.success_rate >= 0` (some steps may fail — that's OK, optimization fixes it)
- No "No EE site for..." errors (correct site mapping)
- No "Body not found" errors (all equipment bodies in scene)
- `simulations/latest.json` saved
- `status.json` updated to `current_phase: "simulate"`

**Checkpoint:**
```bash
cat data/projects/$PROJECT_ID/simulations/latest.json | python -m json.tool | head -20
curl -s http://localhost:8000/api/projects/$PROJECT_ID | python -m json.tool | grep current_phase
```

---

### QA-6: MuJoCo Viewer

**Steps:**
1. On Results page, click "Open MuJoCo 3D Viewer"
2. Viewer window should open and stay open

**Expected:**
- MuJoCo viewer opens and does NOT immediately close
- Scene shows: articulated robot arm (meshes visible, not box), work objects, existing furniture
- If conveyor: belt surface with rollers visible
- Camera housing visible as small dark box
- Viewer stays interactive (orbit, zoom, pan work)

**Checkpoint:**
```bash
# Manual: viewer window stays open, shows meshes
```

---

### QA-7: Optimization (Run Optimization)

**Steps:**
1. On Results page, set max iterations = 3
2. Click "Run Optimization"
3. Wait for iterations to complete

**Expected:**
- Each iteration produces corrected scene (v2.xml, v3.xml, etc.)
- `metrics.success_rate` improves or stays stable across iterations
- No crash during iteration loop
- Iteration history shows corrections applied
- If equipment replaced: new model downloaded, scene rebuilt with real MJCF
- `status.json` updated to `current_phase: "iterate"`

**Checkpoint:**
```bash
ls data/projects/$PROJECT_ID/scenes/v*.xml  # Multiple versions
cat data/projects/$PROJECT_ID/simulations/iteration_history.json | python -m json.tool | head -30
```

---

### QA-8: Dashboard Resume

**Steps:**
1. Navigate to `/` (dashboard)
2. Verify project appears with correct status
3. Click the project card
4. Verify it navigates to the correct step

**Expected:**
- Project listed with latest phase badge
- Click navigates to `/projects/{id}/results`
- All data restored (metrics, iteration history visible)
- Browser reload on any step URL restores state correctly

---

## Regression Checklist

After each phase, run this full regression:

```bash
# Backend
python -m pytest backend/tests/ -x -v
python -m ruff check backend/
python -m ruff format --check backend/

# Frontend
cd frontend && npm run typecheck && npm run lint

# QA sanity (requires running backend + frontend)
curl -s http://localhost:8000/api/projects | python -m json.tool
```

## Known Issues to Watch

| Issue | Symptom | Root Cause | Fix |
|---|---|---|---|
| scene.xml included instead of robot.xml | "Error opening file" in MuJoCo | `find_mjcf_in_dir` returns wrapper | Skip scene.xml, prefer robot-specific XML |
| "No EE site for..." | Simulation fails | Wrong site name mapping | Check actual Menagerie model sites |
| Empty model directories | Robot renders as box geom | Download silently fails | Use `importlib.import_module()` for robot_descriptions |
| Point cloud below grid | Visual bug in Calibrate step | COLMAP Y-down not transformed | Apply (x, -y, -z) transform |
| 500 on simulate | API error | Scene XML invalid (bad include path) | Validate scene with `mujoco.MjModel.from_xml_path()` before returning |
