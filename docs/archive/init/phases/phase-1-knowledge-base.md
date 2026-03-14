# Phase 1 — Knowledge Base + Core Infrastructure

## Goal

Create the equipment catalog (the source of truth for all equipment in the system) and core infrastructure: config, OpenRouter Claude client, prompt loader.

## Tasks

### 1.1 Equipment catalog (`knowledge-base/equipment/`)

Populate with real equipment that has MJCF/URDF models available:

**`manipulators.json`** — robots with Menagerie/URDF models:
- Franka Emika Panda (menagerie: `franka_emika_panda`)
- Universal Robots UR5e (menagerie: `universal_robots_ur5e`)
- ALOHA / Koch v1.1 (menagerie or robot_descriptions)
- xArm (robot_descriptions)
- SO-100 / SO-101

**`conveyors.json`** — conveyor belt modules:
- Generic conveyor segments (custom MJCF — we create simple parametric models)

**`cameras.json`** — camera types:
- Overhead camera
- Microscope camera
- Barcode scanner camera
(These are MuJoCo camera elements, not physical models)

**`fixtures.json`** — static geometry:
- Work tables (various sizes)
- Shelving units
- Storage containers

Each entry follows `EquipmentEntry` schema from Phase 0.

### 1.2 Config (`backend/app/core/config.py`)
- Pydantic `Settings` class with:
  - `OPENROUTER_API_KEY` (from env)
  - `OPENROUTER_MODEL` (default: `anthropic/claude-sonnet-4-20250514`)
  - `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
  - `DATA_DIR`, `MODELS_DIR`, `KNOWLEDGE_BASE_DIR`
  - `MAX_ITERATIONS` (default: 5)

### 1.3 OpenRouter Claude client (`backend/app/core/claude.py`)
- Async client using `httpx` with OpenRouter's OpenAI-compatible endpoint
- Methods:
  - `send_message(system, messages, model) → response` — text completion
  - `send_vision_message(system, images, text, model) → response` — vision with images
- Handles retries (up to 2), error mapping
- Streams are optional (not needed for MVP)

### 1.4 Prompt loader (`backend/app/core/prompts.py`)
- `load_prompt(name: str) → str` — reads from `prompts/` directory
- Prompts are Markdown files with system instructions

### 1.5 Catalog loader (`backend/app/services/catalog.py`)
- `load_equipment_catalog() → dict[str, EquipmentEntry]` — loads all JSONs from knowledge-base
- `validate_equipment_id(equipment_id, catalog)` — raises if not found
- Caches loaded catalog in memory

### 1.6 Tests
- Unit tests for config loading
- Unit test for catalog loader (valid JSON, schema validation)
- Integration test: Claude client sends a simple message via OpenRouter and gets response
- Unit test for prompt loader

## Checkpoint

```bash
# Catalog loads and validates
pytest backend/tests/test_catalog.py -v

# Claude client works via OpenRouter
OPENROUTER_API_KEY=sk-or-... pytest backend/tests/test_claude_client.py -v

# Prompt loader
pytest backend/tests/test_prompts.py -v
```

## Commit
```
feat: equipment catalog, OpenRouter Claude client, core config
```
