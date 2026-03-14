# Plan: Persistent Project State + Dashboard

## Context

The app loses all state on page reload — pipeline step, project data, and simulation results live only in React `useState`. No way to see previous projects or resume work.

**Goal:**
1. Persist project state to disk (`status.json` per project)
2. Reflect current project + step in URL (browser back/forward, direct links)
3. Dashboard listing all projects with statuses — resume any or start new

---

## Feature 1: Backend — ProjectStatus model + service

### What
- New Pydantic model `ProjectStatus` with phase tracking and timestamps
- New service `project_status.py` for CRUD on `status.json` per project
- DRY: extract shared `get_project_dir()` (currently duplicated in capture.py, simulate.py, iterate.py)

### Files
| File | Action |
|---|---|
| `backend/app/models/project.py` | CREATE |
| `backend/app/models/__init__.py` | MODIFY — add exports |
| `backend/app/services/project_status.py` | CREATE |

### Key structures

```python
# models/project.py
PipelinePhase = Literal["upload", "calibrate", "recommend", "build-scene", "simulate", "iterate"]

class PhaseRecord(BaseModel):
    phase: PipelinePhase
    completed_at: datetime

class ProjectStatus(BaseModel):
    id: str
    name: str = ""
    current_phase: PipelinePhase
    created_at: datetime
    updated_at: datetime
    phases_completed: list[PhaseRecord] = []

class ProjectDetail(BaseModel):
    status: ProjectStatus
    dimensions: Dimensions | None = None
    recommendation: Recommendation | None = None
    sim_result: SimResult | None = None
    iteration_history: list[IterationLog] = []
```

```python
# services/project_status.py
def get_project_dir(project_id: str) -> Path
def create_project_status(project_id: str, name: str = "") -> ProjectStatus
def load_project_status(project_id: str) -> ProjectStatus
def advance_phase(project_id: str, phase: PipelinePhase) -> ProjectStatus
def list_all_projects() -> list[ProjectStatus]
def load_project_detail(project_id: str) -> ProjectDetail
```

### Checks
```bash
pytest backend/tests/test_models.py
pytest backend/tests/ -x
```

### Commit
```
feat: add ProjectStatus model and project_status service

Pydantic model for tracking project pipeline phase with timestamps.
Service for CRUD on status.json per project directory.
Centralized get_project_dir replaces duplicated private helpers.
```

---

## Feature 2: Backend — Projects API + integrate status updates

### What
- New router `GET /api/projects` and `GET /api/projects/{id}`
- Integrate `create_project_status` / `advance_phase` calls into existing endpoints
- Replace `_get_project_dir` duplicates with shared import

### Files
| File | Action |
|---|---|
| `backend/app/api/projects.py` | CREATE |
| `backend/app/main.py` | MODIFY — register projects router |
| `backend/app/api/capture.py` | MODIFY — create_project_status + advance_phase, use shared get_project_dir |
| `backend/app/api/recommend.py` | MODIFY — advance_phase |
| `backend/app/api/simulate.py` | MODIFY — advance_phase, use shared get_project_dir |
| `backend/app/api/iterate.py` | MODIFY — advance_phase, use shared get_project_dir |

### Integration points
| Endpoint | Phase |
|---|---|
| `POST /api/capture` | `create_project_status(project_id)` |
| `POST /api/capture/{id}/calibrate` | `advance_phase(id, "calibrate")` |
| `POST /api/recommend` | `advance_phase(id, "recommend")` |
| `POST /{id}/build-scene` | `advance_phase(id, "build-scene")` |
| `POST /{id}/simulate` | `advance_phase(id, "simulate")` |
| `POST /{id}/iterate` | `advance_phase(id, "iterate")` |

### Checks
```bash
pytest backend/tests/ -x
```

### Commit
```
feat: add projects API and integrate status tracking

GET /api/projects returns all projects with statuses.
GET /api/projects/{id} returns full project detail with phase data.
Each pipeline endpoint now updates status.json on completion.
Replaced duplicated _get_project_dir with shared service function.
```

---

## Feature 3: Frontend — Types + API client

### What
- Add TypeScript types mirroring backend ProjectStatus/ProjectDetail
- Add `listProjects()` and `getProject()` API functions

### Files
| File | Action |
|---|---|
| `frontend/src/types/index.ts` | MODIFY — add PipelinePhase, PhaseRecord, ProjectStatus, ProjectDetail |
| `frontend/src/api/client.ts` | MODIFY — add listProjects, getProject |

### Checks
```bash
cd frontend && npm run typecheck
```

### Commit
```
feat: add project status types and API client functions

TypeScript types for PipelinePhase, ProjectStatus, ProjectDetail.
API functions listProjects() and getProject() for dashboard support.
```

---

## Feature 4: Frontend — React Router + Layout

### What
- Enable react-router-dom (already in package.json, unused)
- Set up route structure with Layout wrapper
- Extract header into Layout component with "Back to Dashboard" link

### Routes
```
/                                → Dashboard
/new                             → PhotoUpload (new project)
/projects/:projectId/calibrate   → ProjectWorkflow
/projects/:projectId/recommend   → ProjectWorkflow
/projects/:projectId/simulate    → ProjectWorkflow
/projects/:projectId/results     → ProjectWorkflow
```

### Files
| File | Action |
|---|---|
| `frontend/src/main.tsx` | MODIFY — wrap in BrowserRouter |
| `frontend/src/App.tsx` | REWRITE — router config with Routes |
| `frontend/src/components/Layout.tsx` | CREATE — header + Outlet |

### Checks
```bash
cd frontend && npm run typecheck && npm run build
```

### Commit
```
feat: enable react-router with URL-based navigation

Routes: / (dashboard), /new (upload), /projects/:id/:step (workflow).
Layout component with shared header and dashboard link.
SPA fallback in backend already handles client-side routes.
```

---

## Feature 5: Frontend — Dashboard page

### What
- Landing page listing all projects from `GET /api/projects`
- Card per project: ID/name, current phase badge, timestamps
- Each card links to `/projects/{id}/{step}` to resume
- "New Project" button → `/new`

### Files
| File | Action |
|---|---|
| `frontend/src/pages/Dashboard.tsx` | CREATE |

### Checks
```bash
cd frontend && npm run typecheck && npm run build
```

### Commit
```
feat: add project dashboard with status cards

Lists all projects with phase badges and timestamps.
Click to resume any project at its current step.
New Project button starts fresh upload flow.
```

---

## Feature 6: Frontend — ProjectWorkflow + useProjectState hook

### What
- Extract current App.tsx workflow logic into `ProjectWorkflow.tsx`
- Custom hook `useProjectState` loads project data from backend on mount
- Step guard: redirects to current phase if user navigates beyond it
- Step callbacks use `useNavigate()` to advance URL
- Step progress bar links to completed steps

### Files
| File | Action |
|---|---|
| `frontend/src/pages/ProjectWorkflow.tsx` | CREATE — extracted from App.tsx |
| `frontend/src/hooks/useProjectState.ts` | CREATE — data loading hook |

### Checks
```bash
cd frontend && npm run typecheck && npm run build && npm test
```

### Commit
```
feat: add ProjectWorkflow with persistent state restoration

ProjectWorkflow reads projectId and step from URL.
useProjectState hook loads project data from GET /api/projects/{id}.
Step guard redirects to current phase if URL is ahead of progress.
Browser reload restores full project state from backend.
```

---

## Summary

| # | Feature | Backend | Frontend |
|---|---|---|---|
| 1 | ProjectStatus model + service | models/project.py, services/project_status.py | — |
| 2 | Projects API + status integration | api/projects.py, capture/recommend/simulate/iterate.py | — |
| 3 | Types + API client | — | types/index.ts, api/client.ts |
| 4 | React Router + Layout | — | main.tsx, App.tsx, Layout.tsx |
| 5 | Dashboard page | — | pages/Dashboard.tsx |
| 6 | ProjectWorkflow + state hook | — | pages/ProjectWorkflow.tsx, hooks/useProjectState.ts |
