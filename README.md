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

## Quick start

```bash
git clone https://github.com/user/lang2robo
cd lang2robo
pip install -e ".[training]"   # or -e . without training extras
cd frontend && npm install && npm run build && cd ..
cp .env.example .env           # set ANTHROPIC_API_KEY
uvicorn backend.app.main:app --reload
# Open http://localhost:8000
```

## Environment

```env
ANTHROPIC_API_KEY=sk-ant-...
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
