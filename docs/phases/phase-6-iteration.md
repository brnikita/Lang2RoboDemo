# Phase 6 — Iterative Improvement

## Goal

Implement Module 5: after simulation, Claude analyzes metrics and proposes scene corrections. System applies corrections, re-runs simulation. Loop up to 5 times or until success criteria met.

## Tasks

### 6.1 Iteration service (`backend/app/services/planner.py` — extend)

**`iterate(scene_path, metrics, history, catalog) → Path`**
- Reads current MJCF scene XML
- Sends to Claude via OpenRouter: scene XML + metrics + iteration history + catalog
- System prompt: `prompts/iteration.md`
- Claude returns `SceneCorrections` JSON
- Validates any new equipment IDs against catalog
- Downloads new models if equipment replaced
- Applies corrections to scene → saves new version (`v2.xml`, `v3.xml`...)
- Returns path to corrected scene

### 6.2 Corrections applicator

**`apply_corrections(scene_path, corrections: SceneCorrections) → Path`**
- Position changes: update body positions in MJCF XML
- Add equipment: download model + insert into scene
- Remove equipment: delete body from scene
- Replace equipment: remove old, download new, insert at same position
- Workflow changes: update workflow steps for next simulation
- Saves as new version file, preserving history

### 6.3 Iteration loop

**`run_iteration_loop(project_id, max_iterations=5) → list[IterationLog]`**
- Load initial scene + recommendation
- Run simulation → get metrics
- While not converged and iterations < max:
  - Call `iterate()` for corrections
  - Apply corrections
  - Re-run simulation
  - Log iteration (metrics + corrections)
  - Check stop criteria: `success_rate >= 0.95 and collision_count == 0`
- Return full iteration history

### 6.4 Iteration prompt (`prompts/iteration.md`)
- System prompt: role as robotics optimization engineer
- Input format: MJCF XML, current metrics, history of past iterations
- Output format: `SceneCorrections` JSON schema
- Guidelines: what to adjust (positions, equipment, trajectories)
- Anti-patterns: don't repeat corrections that didn't help

### 6.5 API endpoint (`backend/app/api/iterate.py`)

**`POST /api/projects/{project_id}/iterate`**
- Accepts: `{ max_iterations: int }` (default 5)
- Runs full iteration loop
- Streams progress updates via SSE (Server-Sent Events)
- Returns: final `SimResult` + iteration history

### 6.6 Tests
- Unit test: `apply_corrections` — position change modifies MJCF correctly
- Unit test: stop criteria logic
- Integration test: one iteration cycle with real Claude call → corrections applied → scene valid
- Integration test: full loop (may be slow — mark as `@pytest.mark.slow`)

## Checkpoint

```bash
pytest backend/tests/test_iteration.py -v

# Manual: run iteration loop
curl -X POST http://localhost:8000/api/projects/{id}/iterate \
  -H "Content-Type: application/json" \
  -d '{"max_iterations": 3}'
# → iteration history with improving metrics
```

## Commit
```
feat: iteration module — Claude-driven scene optimization loop
```
