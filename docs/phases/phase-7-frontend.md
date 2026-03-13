# Phase 7 — Frontend

## Goal

Build the React + Three.js web interface covering the full user workflow: photo upload → 3D preview with calibration → scenario input → recommendation review → simulation viewer → iteration dashboard.

## Tasks

### 7.1 Project setup + routing
- Vite + React 19 + TypeScript strict
- React Router: `/upload` → `/calibrate/:id` → `/recommend/:id` → `/simulate/:id`
- API client module (fetch wrapper with base URL, error handling)
- Shared layout with step indicator (Upload → Calibrate → Plan → Simulate)

### 7.2 PhotoUpload component (`components/PhotoUpload.tsx`)
- Drag-and-drop zone for 10-30 photos
- Photo thumbnails with remove button
- Upload button → POST /api/capture (multipart)
- Progress bar during DISCOVERSE reconstruction
- On success → navigate to calibration page

### 7.3 SceneViewer3D component (`components/SceneViewer3D.tsx`)
- Three.js scene using @react-three/fiber
- Loads reconstructed mesh (GLB/OBJ from DISCOVERSE)
- Orbit controls for rotation/zoom
- **Calibration mode**: user clicks two points on mesh, enters real distance
- Displays dimensions and detected zones after calibration
- POST /api/capture/{id}/calibrate on submit

### 7.4 FloorPlanEditor component (`components/FloorPlanEditor.tsx`)
- 2D top-down view of SpaceModel
- Shows zones (colored polygons), doors, windows, existing equipment
- Editable: drag zones, adjust boundaries
- "Confirm" button → proceeds to recommendation

### 7.5 RecommendationView component (`components/RecommendationView.tsx`)
- Left panel: text plan from Claude (markdown rendered)
- Right panel: 3D scene with equipment placed (Three.js)
  - Room mesh as background
  - Equipment shown as colored bounding boxes at recommended positions
  - Labels for each piece of equipment
- Scenario text input for modifications
- Buttons: "Confirm Plan" / "Modify" (re-sends to Claude)
- On confirm → POST /api/projects/{id}/build-scene

### 7.6 SimulationPlayer component (`components/SimulationPlayer.tsx`)
- Receives rendered frames from backend via WebSocket
- Playback controls: play, pause, step forward/back, speed
- Current step highlight in workflow list sidebar
- Alternative: Three.js replay of recorded trajectories

### 7.7 MetricsDashboard component (`components/MetricsDashboard.tsx`)
- Displays SimMetrics: cycle time, success rate, collision count
- Per-step breakdown table (success/fail, duration, errors)
- Iteration history chart (metrics over iterations)
- "Run Iteration" button → triggers optimization loop
- Real-time updates during iteration (SSE)
- Stop button for manual halt

### 7.8 API client (`src/api/client.ts`)
- Typed functions for every backend endpoint
- Request/response types from `types/index.ts`
- Error handling with user-friendly messages
- SSE helper for iteration streaming

### 7.9 Tests
- Component render tests (vitest + @testing-library/react)
- API client unit tests with mock responses
- Type-checking: `tsc --noEmit` passes

## Checkpoint

```bash
cd frontend && npm run build  # production build succeeds
cd frontend && npx tsc --noEmit  # no type errors
cd frontend && npm test  # component tests pass

# Manual: open http://localhost:5173, walk through full flow
```

## Commit
```
feat: frontend — React UI with 3D preview, plan editor, simulation viewer
```
