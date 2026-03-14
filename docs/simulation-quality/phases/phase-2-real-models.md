# Phase 2 — Real Equipment Models in Scenes

## Goal

Replace box-geom placeholders with real MJCF models in generated scenes. Manipulators get full articulated mesh from MuJoCo Menagerie. Conveyors get parametric MJCF with belt surface and slide joint. Cameras get visual housing bodies. Fixtures stay as box geoms.

## Tasks

### 2.1 Ensure models are downloaded (`backend/app/services/downloader.py`)

**`download_equipment_model(equipment_id: str) → Path`**
- Verify `robot_descriptions` MJCF path resolution works for all 8 catalog manipulators
- Fix empty model directories — ensure actual files are copied/symlinked to `models/{equipment_id}/`
- Add `_verify_model_dir(path: Path) → bool` — checks dir contains at least one `.xml` file

**`_find_mjcf_in_dir(model_dir: Path) → Path | None`**
- Search for main MJCF entry point (e.g., `xarm7.xml`, `panda.xml`, `scene.xml`)
- Return path to the root MJCF file

### 2.2 Parametric conveyor MJCF generator (`backend/app/services/scene.py`)

**`_generate_conveyor_mjcf(equipment_id: str, specs: dict) → str`**
- Generate MJCF XML string for a conveyor with:
  - Belt surface: flat box geom with high friction (`friction="1 0.005 0.0001"`)
  - Slide joint along belt axis (`type="slide"`, range based on `length_m`)
  - Side rails: two thin box geoms
  - End rollers: two cylinder geoms
- Dimensions from catalog specs: `length_m`, `width_m`, `height_m`
- Returns inline XML string (no file, embedded in scene)

### 2.3 Include real manipulator models (`backend/app/services/scene.py`)

**`_add_manipulator_to_scene(root, placement, model_dir, scene_dir) → None`**
- Find MJCF entry point in `model_dir`
- Compute relative path from scene directory to model MJCF
- Create positioning `<body>` with `pos` and `euler` from placement
- Add `<include file="relative/path/to/robot.xml"/>` inside positioning body
- Add end-effector `<site>` if not already in the model (for IK target in Phase 3)

**`_add_conveyor_to_scene(root, placement, specs) → None`**
- Generate conveyor MJCF via `_generate_conveyor_mjcf()`
- Parse XML string and append to worldbody at placement position
- Add velocity actuator to `<actuator>` section

**`_add_camera_body_to_scene(root, placement, specs) → None`**
- Keep existing `<camera>` element (FOV, position)
- Add small visual body (dark box) representing camera housing

### 2.4 Update scene generator dispatch (`backend/app/services/scene.py`)

**Modify `generate_mjcf_scene()`:**
- Route by equipment type:
  - `"manipulator"` → `_add_manipulator_to_scene()`
  - `"conveyor"` → `_add_conveyor_to_scene()`
  - `"camera"` → `_add_camera_body_to_scene()`
  - `"fixture"` → existing box geom approach (unchanged)
- Pass `scene_dir` (output directory) so relative paths can be computed

### 2.5 Tests

- Unit test: `_generate_conveyor_mjcf()` produces valid XML with joint + actuator
- Unit test: `_find_mjcf_in_dir()` finds entry point in Menagerie structure
- Integration test: build scene with `ufactory_xarm7` → MuJoCo loads successfully, `model.njnt > 5`
- Integration test: build scene with `conveyor_500mm` → MuJoCo loads, has slide joint + velocity actuator
- Integration test: build scene with no manipulator (pickup point scenario) → MuJoCo loads, `model.njnt >= 1` (conveyor only)
- Scenario test: build scene for each of the 4 SPEC scenarios → all load in MuJoCo without errors

## Checkpoint

```bash
pytest backend/tests/test_scene.py -v
pytest backend/tests/test_downloader.py -v

# Manual: build scene for 3D print farm scenario
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('data/projects/{id}/scenes/v1.xml')
print(f'Bodies: {m.nbody}, Joints: {m.njnt}, Actuators: {m.nu}, Geoms: {m.ngeom}')
# Expect: njnt > 7 (robot joints), nu > 0 (actuators), ngeom > 10 (meshes)
"

# Manual: open in MuJoCo viewer → see articulated robot arm with mesh
```

## Commit
```
feat: include real MJCF models for manipulators and parametric conveyors
```
