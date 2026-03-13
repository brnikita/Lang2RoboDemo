# Phase 8 — Integration & Deployment

## Goal

Wire everything end-to-end. Docker Compose setup. Smoke test with a real scenario. Final polish.

## Tasks

### 8.1 End-to-end pipeline test
- Integration test that runs the full pipeline:
  1. Upload sample photos → project created
  2. Calibrate → SpaceModel
  3. Submit scenario → Recommendation
  4. Build scene → valid MJCF
  5. Simulate → SimResult with metrics
  6. Iterate (1-2 rounds) → improved metrics
- Uses real Claude via OpenRouter, real MuJoCo, real DISCOVERSE
- Mark as `@pytest.mark.e2e` (slow, requires API key)

### 8.2 Docker Compose (`docker-compose.yml`)
- **backend** service: Python + FastAPI + all deps (mujoco, discoverse, etc.)
- **frontend** service: Node build → nginx serve
- Shared volume for `data/` and `models/`
- Environment variables from `.env`
- Health checks

### 8.3 Backend static serving
- FastAPI serves built frontend from `frontend/dist/`
- Single deployment: `uvicorn backend.app.main:app` serves both API and UI
- Fallback route for SPA (client-side routing)

### 8.4 WebSocket for simulation frames
- Backend renders MuJoCo frames → encodes as JPEG → sends via WebSocket
- Frontend SimulationPlayer consumes WebSocket stream
- Handles connection lifecycle, reconnection

### 8.5 SSE for iteration progress
- Backend streams iteration updates (current iteration, metrics, status)
- Frontend MetricsDashboard consumes SSE
- Graceful handling of client disconnect

### 8.6 Error handling & edge cases
- API error responses with meaningful messages
- Frontend error boundaries
- Timeout handling for long operations (DISCOVERSE, simulation)
- File size limits for photo upload
- Cleanup of orphaned project data

### 8.7 Environment setup docs
- `.env.example` with all required variables
- Installation instructions in existing README or inline comments
- Platform-specific notes (DISCOVERSE may need specific build tools)

### 8.8 Smoke test with real scenarios
Test with at least 2 scenarios from spec:
1. **3D print farm**: robot + conveyor + camera
2. **PVZ (pickup point)**: conveyor + camera, no robot

Verify complete flow produces reasonable results.

## Checkpoint

```bash
# E2E test
OPENROUTER_API_KEY=sk-or-... pytest backend/tests/test_e2e.py -v --timeout=300

# Docker
docker compose up --build
# Open http://localhost:8000 → full demo works

# Smoke test: upload photos, enter scenario, get simulation results
```

## Commit
```
feat: end-to-end integration, Docker Compose, smoke tests
```
