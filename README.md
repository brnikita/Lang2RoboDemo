# Lang2Robo

**Text description → robotic cell simulation → iterative improvement.** No real hardware. Works with any type of small business and any set of equipment — with or without robots.

```
Room photos + scenario text
  → 3D scene reconstruction
  → AI proposes a robotization plan (text + diagram)
  → User confirms
  → Auto-download of models
  → Prototype assembly in MuJoCo
  → Runs and iterative policy improvement
```

## Features

- **Space capture** — Upload 10–30 photos; DISCOVERSE reconstructs the room to MuJoCo (MJCF). Claude Vision extracts zones, equipment, doors, windows.
- **AI recommendation** — Describe your automation scenario in text; Claude returns a robotization plan (equipment from a strict catalog, workflow steps, targets).
- **Scene assembly** — Auto-download MJCF/URDF from MuJoCo Menagerie / catalog; assemble room + robots + work objects into one MuJoCo scene.
- **Simulation** — Scripted IK for manipulators, conveyor belt physics, camera inspection. Metrics: cycle time, success rate, collisions.
- **Iterative improvement** — Claude analyzes metrics and suggests corrections (positions, equipment swap); up to 5 iterations until success.
- **Policy training (MVP v2)** — Record scripted demos, fine-tune SmolVLA with LeRobot when manipulators are present.

## Stack

| Layer           | Technology                          |
|----------------|-------------------------------------|
| Simulator      | MuJoCo (CPU-only, 4000× realtime)   |
| Robot models   | MuJoCo Menagerie + robot_descriptions |
| 3D reconstruction | DISCOVERSE (photos → MuJoCo)      |
| AI planning    | Claude API (Vision + text)          |
| Backend        | FastAPI, Pydantic                    |
| Frontend       | React, TypeScript, Three.js          |
| Policy training| LeRobot, SmolVLA (450M)              |

**Minimum:** Python 3.11+, 8 GB RAM, any CPU. GPU not required.

## Environment

Copy `.env.example` and fill in your OpenRouter API key:

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4.6    # optional, default
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional, default
```

---

## Development mode

Two processes: backend with hot-reload + frontend dev server with HMR.

**1. Install dependencies:**

```bash
pip install -e ".[dev]"
cd frontend && npm install && cd ..
```

**2. Run backend (terminal 1):**

```bash
python -m uvicorn backend.app.main:app --reload
```

Backend starts at **http://localhost:8000** (API under `/api`).

**3. Run frontend (terminal 2):**

```bash
cd frontend
npm run dev
```

Frontend starts at **http://localhost:5173** with API proxy to `localhost:8000`.

Open **http://localhost:5173** in the browser.

---

## Production mode (Docker)

Single command — builds both backend and frontend, serves everything from one container.

```bash
docker compose up --build
```

App available at **http://localhost:8000**.

Subsequent launches without code changes:

```bash
docker compose up
```

> If you changed code, add `--build` to rebuild the image.

Stop:

```bash
docker compose down
```

## Project layout

```
lang2robo/
├── backend/          # FastAPI app, API routes, services, Pydantic models
├── frontend/         # React + Three.js UI
├── knowledge-base/  # Equipment catalog (JSON)
├── prompts/         # System prompts for Claude
├── data/             # Per-project photos, reconstruction, scenes, simulations
└── SPEC.md           # Full specification
```

## License

MIT — see [LICENSE](LICENSE).
