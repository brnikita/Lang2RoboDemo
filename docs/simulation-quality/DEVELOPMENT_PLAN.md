# Simulation Quality — Development Plan

## Overview

Upgrade the simulation engine from placeholder boxes to presentation-quality visuals. After this work, the MuJoCo viewer shows articulated robots, moving conveyors, and real pick-place-transport workflows for any scenario Claude generates.

Build order: **point cloud fixes → real models → controllers → visual runner → iteration visibility**.
Every phase produces a runnable, testable increment. Each phase = commit + push.

All code follows CLAUDE.md: DRY, SOLID, max 20 lines per method, Pydantic models, Google docstrings, `__all__` exports, TDD.

### Phase Dependency Graph

```
Phase 1  Point Cloud Quality (coordinate fix, rescale, density, grid)
   ↓
Phase 2  Real Equipment Models (Menagerie include, parametric conveyor, camera body)
   ↓
Phase 3  Controllers (IK engine, grasp manager — reusable for any manipulator)
   ↓
Phase 4  Action Executors (manipulator/conveyor/camera executors using controllers)
   ↓
Phase 5  Visual Simulation Runner (viewer plays full workflow in real-time)
   ↓
Phase 6  Iteration Visibility (model replacement rebuild, before/after comparison)
   ↓
Phase 7  QA End-to-End Testing (real room photos, all steps verified)
```

### Phase Summary

| Phase | Deliverable | Checkpoint |
|-------|-------------|------------|
| [Phase 1](./phases/phase-1-pointcloud.md) | Correct point cloud orientation, calibration rescale, denser reconstruction, adaptive grid | Point cloud renders correctly in SceneViewer3D |
| [Phase 2](./phases/phase-2-real-models.md) | Real MJCF models for manipulators, parametric conveyors, camera housings | MuJoCo viewer shows meshes + joints instead of boxes |
| [Phase 3](./phases/phase-3-controllers.md) | Jacobian IK engine, weld-based grasp manager — universal for any manipulator | IK reaches target pose; grasp attaches/detaches objects |
| [Phase 4](./phases/phase-4-executors.md) | Action executors: pick/place/move, transport, inspect — using controllers | Full workflow runs with real physics for any scenario |
| [Phase 5](./phases/phase-5-visual-runner.md) | Visual simulation mode — viewer plays workflow in real-time | "Launch Viewer" → watch automation cycle |
| [Phase 6](./phases/phase-6-iteration-visibility.md) | Iteration loop rebuilds scene on equipment swap, metrics visibly improve | "Run Optimization" → visible improvements across iterations |
| [Phase 7](./phases/phase-7-qa-testing.md) | End-to-end QA with real room photos (16 photos of home office) | Upload → Calibrate → Plan → Build → Simulate → Viewer → Optimize → Resume — all pass |

### Scenario Coverage

Every phase is validated against multiple SPEC scenarios:

| Scenario | Robot | Conveyor | Camera | Phases exercised |
|---|---|---|---|---|
| 3D print farm | xArm7 | 500mm | overhead | All |
| Pickup point | NONE | 1000mm | barcode | 1, 2 (conveyor only), 4b, 5 |
| Electronics repair | WidowX→Franka | NONE | microscope | 1, 2 (robot only), 3, 4a, 5, 6 (replacement) |
| Dark kitchen | Franka | 500mm | overhead | All |
