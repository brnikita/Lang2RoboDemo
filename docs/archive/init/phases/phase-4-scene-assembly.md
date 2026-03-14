# Phase 4 — Scene Assembly

## Goal

Implement Module 3: after user confirms recommendation, download real robot/equipment models and compose a complete MJCF scene file that MuJoCo can load.

## Tasks

### 4.1 Model downloader (`backend/app/services/downloader.py`)

**`download_equipment_models(placements: list[EquipmentPlacement]) → dict[str, Path]`**
- For each placement, look up `MjcfSource` in catalog
- If `menagerie_id` → fetch from MuJoCo Menagerie (pip package or git)
- If `robot_descriptions_id` → fetch via `robot_descriptions` package
- If `urdf_url` → download URDF, convert to MJCF using `urdf2mjcf` / trimesh
- Cache downloaded models in `models/` directory (skip if already cached)
- Returns mapping: equipment_id → local path to MJCF file

### 4.2 MJCF scene builder (`backend/app/services/scene.py`)

**`generate_mjcf_scene(space, recommendation, models, output_path) → Path`**
- Start with DISCOVERSE MJCF as base (room geometry)
- Add existing equipment as static bodies (simplified box collision shapes)
- Include each robot/equipment model via MJCF `<include>` or inline, positioned per recommendation
- Add work objects as dynamic bodies (box/cylinder/sphere with mass)
- Set up cameras, lights, ground plane
- Save complete MJCF to `data/projects/{id}/scenes/v1.xml`

**`add_equipment_to_scene(mjcf_root, model_path, position, orientation)`**
- Inserts equipment MJCF into scene at specified pose
- Handles coordinate transforms, naming conflicts

**`add_dynamic_body(mjcf_root, name, shape, size, mass, position)`**
- Creates MuJoCo body with geom + freejoint for graspable objects

### 4.3 MJCF validation

**`validate_mjcf(scene_path: Path) → bool`**
- Attempts `mujoco.MjModel.from_xml_path()` — if it loads, scene is valid
- Reports specific XML errors if it fails

### 4.4 API endpoint (`backend/app/api/simulate.py` — scene part)

**`POST /api/projects/{project_id}/build-scene`**
- Accepts: `{ recommendation_id: str }` (or uses latest)
- Downloads models, builds MJCF scene
- Validates scene loads in MuJoCo
- Returns: scene metadata (path, body count, validation status)

### 4.5 Tests
- Unit test: `add_dynamic_body` creates valid MuJoCo XML fragment
- Unit test: `generate_mjcf_scene` with minimal inputs → valid MJCF
- Integration test: download a real Menagerie model (e.g., Franka) → file exists
- Integration test: build scene with real model → `mujoco.MjModel.from_xml_path()` succeeds

## Checkpoint

```bash
pytest backend/tests/test_downloader.py -v
pytest backend/tests/test_scene.py -v

# Manual: build scene and verify MuJoCo loads it
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('data/projects/test/scenes/v1.xml')
print(f'Bodies: {m.nbody}, Joints: {m.njnt}, Geoms: {m.ngeom}')
"
```

## Commit
```
feat: scene assembly — model download + MJCF composition
```
