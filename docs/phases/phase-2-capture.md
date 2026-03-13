# Phase 2 — Capture Module

## Goal

Implement Module 1 from spec: photos → 3D reconstruction (DISCOVERSE) → scene analysis (Claude Vision) → SpaceModel. Plus the API endpoint and calibration logic.

## Tasks

### 2.1 DISCOVERSE integration (`backend/app/services/vision.py`)

**`reconstruct_scene(photos_dir: Path, output_dir: Path) → SceneReconstruction`**
- Calls DISCOVERSE `real2sim` pipeline: photos → COLMAP SfM → 3DGS → MJCF export
- Returns `SceneReconstruction` with paths to mesh, MJCF, point cloud

**`calibrate_scale(reconstruction, calibration: ReferenceCalibration) → SceneReconstruction`**
- Computes scale factor from two user-picked points + known real distance
- Applies scale to mesh and MJCF
- Updates dimensions in reconstruction

> Note: DISCOVERSE installation may require specific build steps. Document any platform-specific issues.

### 2.2 Claude Vision analysis (`backend/app/services/vision.py`)

**`analyze_scene(photos: list[Path], reconstruction: SceneReconstruction) → SceneAnalysis`**
- Sends photos + reconstruction dimensions to Claude via OpenRouter
- System prompt: `prompts/vision_analysis.md`
- Claude returns structured JSON: zones, existing equipment, doors, windows
- Parses response into `SceneAnalysis` model

### 2.3 SpaceModel composition

**`build_space_model(reconstruction, analysis) → SpaceModel`**
- Merges DISCOVERSE reconstruction data with Claude Vision analysis
- Simple composition — no complex logic

### 2.4 Vision analysis prompt (`prompts/vision_analysis.md`)
- System prompt instructing Claude to analyze room photos
- Output format: JSON matching `SceneAnalysis` schema
- Instructions: identify zones, equipment, doors, windows, estimate positions

### 2.5 API endpoint (`backend/app/api/capture.py`)

**`POST /api/capture`**
- Accepts: multipart file upload (10-30 photos)
- Creates project directory under `data/projects/{uuid}/`
- Saves photos, runs DISCOVERSE, returns project_id + reconstruction preview data

**`POST /api/capture/{project_id}/calibrate`**
- Accepts: `ReferenceCalibration` JSON body
- Applies scale calibration
- Runs Claude Vision analysis
- Returns: `SpaceModel` JSON

### 2.6 Tests
- Unit test: `calibrate_scale` with known geometry (two points, known distance)
- Unit test: `build_space_model` composition
- Integration test: `analyze_scene` with sample photos → valid SceneAnalysis
- API test: upload endpoint creates project directory, returns ID

## Checkpoint

```bash
# Run tests
pytest backend/tests/test_vision.py -v
pytest backend/tests/test_capture_api.py -v

# Manual E2E: upload real photos, get SpaceModel
curl -X POST http://localhost:8000/api/capture \
  -F "photos=@photo1.jpg" -F "photos=@photo2.jpg" ...
# → returns project_id

curl -X POST http://localhost:8000/api/capture/{id}/calibrate \
  -H "Content-Type: application/json" \
  -d '{"point_a": [0,0,0], "point_b": [1,0,0], "real_distance_m": 0.9}'
# → returns SpaceModel JSON
```

## Commit
```
feat: capture module — DISCOVERSE reconstruction + Claude Vision analysis
```
