# Lang2Robo MVP — Development Plan

## Overview

Build order: **contracts → core infra → modules bottom-up → frontend → integration**.
Every phase produces a runnable, testable increment. Each phase = commit + push.

All integrations are **real** — no stubs. Claude API is accessed via **OpenRouter**.

### Phase Dependency Graph

```
Phase 0  Skeleton + Data Models
   ↓
Phase 1  Knowledge Base + Core Infrastructure
   ↓
Phase 2  Capture Module (DISCOVERSE + Claude Vision)
   ↓
Phase 3  Recommendation Module (Claude via OpenRouter)
   ↓
Phase 4  Scene Assembly (Menagerie download + MJCF build)
   ↓
Phase 5  Simulation (MuJoCo scripted IK)
   ↓
Phase 6  Iterative Improvement (Claude + re-simulation loop)
   ↓
Phase 7  Frontend (React + Three.js + full UI)
   ↓
Phase 8  Integration (end-to-end, Docker, polish)
```

### Phase Summary

| Phase | Deliverable | Checkpoint |
|-------|-------------|------------|
| [Phase 0](./phases/phase-0-skeleton.md) | Project structure, all Pydantic + TS types, pyproject.toml, package.json | `pytest` passes, `tsc --noEmit` passes |
| [Phase 1](./phases/phase-1-knowledge-base.md) | Equipment catalog JSONs, config, OpenRouter Claude client, prompt loader | Unit tests pass, client can call Claude via OpenRouter |
| [Phase 2](./phases/phase-2-capture.md) | DISCOVERSE Real2Sim integration, Claude Vision scene analysis, calibration | Photos → reconstructed MJCF + SpaceModel JSON |
| [Phase 3](./phases/phase-3-recommendation.md) | Claude generates equipment plan from SpaceModel + scenario text | Scenario text → validated Recommendation JSON with real catalog IDs |
| [Phase 4](./phases/phase-4-scene-assembly.md) | Model download from Menagerie/URDF, MJCF scene composition | Recommendation → loadable MJCF with all bodies |
| [Phase 5](./phases/phase-5-simulation.md) | MuJoCo scripted simulation, IK controller, conveyor/camera executors | MJCF scene → SimResult with metrics |
| [Phase 6](./phases/phase-6-iteration.md) | Claude analyzes metrics, proposes corrections, re-runs simulation | 3-5 iteration loop, success_rate improves |
| [Phase 7](./phases/phase-7-frontend.md) | React app: photo upload, 3D preview, plan editor, simulation viewer | Full UI workflow in browser |
| [Phase 8](./phases/phase-8-integration.md) | End-to-end pipeline, Docker Compose, env setup, smoke test | `docker compose up` → full demo works |

### Claude API via OpenRouter

All Claude calls go through OpenRouter (`https://openrouter.ai/api/v1`).
Environment variables:
```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514
```

The Claude client wrapper uses the OpenAI-compatible API format that OpenRouter provides.

### External Dependencies (all real)

| Dependency | Purpose | Install |
|------------|---------|---------|
| DISCOVERSE | Photos → MuJoCo scene (Real2Sim) | `pip install discoverse` |
| MuJoCo | Physics simulation | `pip install mujoco` |
| MuJoCo Menagerie | Robot MJCF models | `pip install mujoco-menagerie` |
| robot_descriptions | Additional URDF/MJCF | `pip install robot-descriptions` |
| OpenRouter | Claude API access | HTTP client (httpx) |
| LeRobot + SmolVLA | Policy training (MVP v2) | Future phase |
