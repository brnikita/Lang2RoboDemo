# Phase 0 — Project Skeleton + Data Models

## Goal

Set up the entire project structure and define **all data contracts** (Pydantic models, TypeScript types). No business logic yet — only types, project config, and empty module files with `__all__` exports.

This is the foundation. Every later phase imports these models. Getting them right here prevents mismatches between modules.

## Tasks

### 0.1 Python project setup
- `pyproject.toml` with all real dependencies (fastapi, uvicorn, pydantic, mujoco, discoverse, anthropic, httpx, trimesh, numpy, pillow)
- `backend/` package structure matching SPEC.md
- `prompts/` directory (empty .md files)
- `data/` and `models/` directories
- `.env.example` with `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- `.gitignore` (Python, Node, data/, models/, .env)

### 0.2 All Pydantic models (`backend/app/models/`)
Files and their models (all from SPEC.md):

**`space.py`**:
- `ReferenceCalibration` — point_a, point_b, real_distance_m
- `Dimensions` — width_m, length_m, ceiling_m, area_m2
- `Zone` — name, polygon, area_m2
- `Door` — position, width_m
- `Window` — position, width_m
- `ExistingEquipment` — name, category, position, confidence
- `SceneReconstruction` — mesh_path, mjcf_path, pointcloud_path, dimensions
- `SceneAnalysis` — zones, existing_equipment, doors, windows
- `SpaceModel` — dimensions, zones, existing_equipment, doors, windows, reconstruction

**`recommendation.py`**:
- `EquipmentPlacement` — equipment_id, position, orientation_deg, purpose, zone
- `WorkObject` — name, shape, size, mass_kg, position, count
- `WorkflowStep` — order, action, equipment_id, target, duration_s, params
- `ExpectedMetrics` — (cycle_time_s, throughput, etc.)
- `Recommendation` — equipment, work_objects, target_positions, workflow_steps, expected_metrics

**`simulation.py`**:
- `StepResult` — success, duration_s, collision_count, error
- `SimMetrics` — cycle_time_s, success_rate, collision_count, failed_steps
- `SimResult` — steps, metrics

**`iteration.py`**:
- `PositionChange` — equipment_id, new_position, new_orientation_deg
- `EquipmentReplacement` — old_equipment_id, new_equipment_id, reason
- `SceneCorrections` — position_changes, add_equipment, remove_equipment, replace_equipment, workflow_changes
- `IterationLog` — iteration, metrics, corrections_applied

**`equipment.py`**:
- `MjcfSource` — menagerie_id, robot_descriptions_id, urdf_url
- `PlacementRules` — min_zone_m2, constraints
- `EquipmentEntry` — id, name, type, specs, mjcf_source, price_usd, purchase_url, placement_rules

### 0.3 Frontend setup
- `frontend/package.json` (react 19, three, @react-three/fiber, @react-three/drei, typescript, vite)
- `frontend/tsconfig.json` (strict mode, no `any`)
- `frontend/src/types/index.ts` — TypeScript interfaces mirroring all Pydantic models
- `frontend/src/App.tsx` — empty shell
- `frontend/src/vite-env.d.ts`

### 0.4 Empty service/API modules
- `backend/app/services/` — empty `__init__.py` files for vision, planner, scene, simulator, downloader
- `backend/app/api/` — empty route files for capture, recommend, simulate, iterate
- `backend/app/core/` — empty config.py, claude.py, prompts.py
- `backend/app/main.py` — FastAPI app with CORS, no routes yet

### 0.5 Tests skeleton
- `backend/tests/conftest.py`
- Empty test files matching service modules
- `frontend/vitest.config.ts`

## Checkpoint

```bash
# Python
cd backend && pip install -e ".[dev]" && pytest --collect-only  # all test files discovered
python -c "from app.models.space import SpaceModel; print('OK')"
python -c "from app.models.recommendation import Recommendation; print('OK')"
python -c "from app.models.simulation import SimResult; print('OK')"

# Frontend
cd frontend && npm install && npx tsc --noEmit  # types compile
```

## Commit
```
feat: project skeleton with all Pydantic models and TypeScript types
```
