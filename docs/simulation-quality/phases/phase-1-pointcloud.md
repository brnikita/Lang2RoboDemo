# Phase 1 — Point Cloud Quality

## Goal

Fix the 3D point cloud so it renders correctly in the Calibrate step: properly oriented, matching the room scale after calibration, dense enough to be recognizable, with a grid that adapts to the scene.

## Tasks

### 1.1 Coordinate transform (`backend/app/services/reconstruction.py`)

**`_export_pointcloud(points3D, colors, output_path) → Path`**
- Transform COLMAP coordinates (X-right, Y-down, Z-forward) to Three.js convention (X-right, Y-up, Z-back)
- Apply: `(x, y, z) → (x, -y, -z)`
- Export as binary PLY via trimesh

### 1.2 Rescale point cloud on calibration (`backend/app/services/reconstruction.py`)

**`rescale_pointcloud(ply_path: Path, scale_factor: float) → None`**
- Load PLY via trimesh
- Multiply all vertices by `scale_factor`
- Re-export to same path
- Called from `calibrate_scale()` alongside existing mesh/MJCF rescaling

### 1.3 Increase reconstruction density (`backend/app/services/reconstruction.py`)

Update SfM quality parameters in `reconstruct_scene()`:
- `max_num_features`: 8192 → 32768
- `max_ratio`: 0.9 → 0.8 (stricter feature matching)
- `min_num_matches`: 10 → 15 (reduce noise)
- Add sequential matching for adjacent photos (improves coverage for walk-around photo sets)

### 1.4 Dynamic grid in viewer (`frontend/src/components/SceneViewer3D.tsx`)

**Update `PointCloud` component:**
- After loading geometry, compute bounding box extents
- Set grid size = `max(extentX, extentZ) * 1.5` (clamped to `[1, 50]`)
- Grid divisions = 20 (fixed)
- Center grid at point cloud centroid projected to Y=0

### 1.5 Tests

- Unit test: coordinate transform produces Y-up output from Y-down input
- Unit test: `rescale_pointcloud` scales vertices correctly
- Unit test: bounding box computation for grid sizing
- Integration test: `reconstruct_scene()` with test images → PLY with > 1000 vertices

## Checkpoint

```bash
pytest backend/tests/test_reconstruction.py -v

# Manual: upload photos → Calibrate step shows point cloud above grid,
# correctly oriented (floor at bottom, ceiling at top)
# After calibration: point cloud shrinks to room scale, grid matches
```

## Commit
```
fix: point cloud coordinate transform, calibration rescale, denser reconstruction
```
