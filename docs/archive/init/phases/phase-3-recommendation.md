# Phase 3 — Recommendation Module

## Goal

Implement Module 2: SpaceModel + user scenario text → Claude generates equipment plan → validated Recommendation with real catalog IDs.

## Tasks

### 3.1 Recommendation service (`backend/app/services/planner.py`)

**`generate_recommendation(space: SpaceModel, scenario: str) → Recommendation`**
- Loads equipment catalog
- Sends SpaceModel JSON + scenario + full catalog to Claude via OpenRouter
- System prompt: `prompts/recommendation.md`
- Claude returns structured JSON: equipment placements, work objects, workflow steps, target positions
- **Validation**: every `equipment_id` is checked against catalog. If invalid — retry (up to 2 times)
- Prices come from catalog, not from Claude's response

### 3.2 Response parser + validator

**`parse_and_validate(response_text: str, catalog) → Recommendation`**
- Extracts JSON from Claude response (handles markdown code blocks if present)
- Validates against `Recommendation` Pydantic model
- Cross-checks all `equipment_id` references against catalog
- Cross-checks `WorkflowStep.equipment_id` against `Recommendation.equipment` list
- Validates `WorkflowStep.target` keys exist in `target_positions`
- On validation failure: re-prompts Claude with error details (up to 2 retries)

### 3.3 Context formatter

**`format_recommendation_context(space, scenario, catalog) → str`**
- Formats SpaceModel as concise JSON
- Includes full equipment catalog (filtered by relevance if needed)
- Includes scenario text
- Structured so Claude can reference equipment IDs directly

### 3.4 Recommendation prompt (`prompts/recommendation.md`)
- System prompt for Claude: role, output format, constraints
- Must reference equipment only from provided catalog
- Output schema: matches `Recommendation` model
- Include examples of good output format

### 3.5 API endpoint (`backend/app/api/recommend.py`)

**`POST /api/recommend`**
- Accepts: `{ project_id: str, scenario: str }`
- Loads SpaceModel from project data
- Calls `generate_recommendation`
- Saves recommendation to `data/projects/{id}/recommendation/`
- Returns: `Recommendation` JSON + text plan

### 3.6 Tests
- Unit test: `parse_and_validate` with valid/invalid JSON
- Unit test: cross-validation catches invalid equipment_id
- Integration test: real Claude call with sample SpaceModel + scenario → valid Recommendation
- API test: POST /recommend with project_id → Recommendation JSON

## Checkpoint

```bash
pytest backend/tests/test_planner.py -v
pytest backend/tests/test_recommend_api.py -v

# Manual: real scenario
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"project_id": "...", "scenario": "3D print farm, robot picks finished prints..."}'
# → Recommendation JSON with real equipment IDs from catalog
```

## Commit
```
feat: recommendation module — Claude generates validated equipment plans
```
