# Lang2Robo — Demo MVP Specification

Platform: text description of a business process → simulation of an automated cell → iterative improvement. No real hardware. Works with any type of small business and any set of equipment — with or without robots.

```
Room photos + scenario text
  → 3D scene reconstruction
  → AI proposes a robotization plan (text + diagram)
  → User confirms
  → Auto-download of models, SDK
  → Prototype assembly in MuJoCo
  → Runs and iterative policy improvement
```

---

## MVP Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Simulator | **MuJoCo** | `pip install mujoco`, CPU-only, 4000× realtime, Python API |
| Robot models | **MuJoCo Menagerie** + `robot_descriptions` | 135+ ready MJCF/URDF, selected from knowledge-base |
| Equipment catalog | **knowledge-base/equipment/** (JSON) | Claude selects only from the real catalog, does not invent |
| 3D reconstruction | **DISCOVERSE** | Open-source Real2Sim pipeline (photos → MuJoCo), MIT, 650 FPS render |
| Policy training | **LeRobot** + **SmolVLA** (450M) | Runs on MacBook, 30Hz, training in sim (LIBERO/Meta-World) |
| AI planning | **Claude API** | Vision for photos, text for recommendations and iterations |
| Orchestration | **asyncio** | Zero extra dependencies, linear pipeline |
| Backend | **FastAPI** + **Pydantic** | API + validation |
| Frontend | **React** + **TypeScript** + **Three.js** | 3D scene preview, plan editor |
| Model conversion | `urdf2mjcf`, `robot-format-converter` | URDF ↔ MJCF ↔ SDF |

**Minimum requirements**: Python 3.11+, 8 GB RAM, any CPU. GPU not required.

---

## Module 1. Space capture

### Input

The user uploads 10–30 photos of the space (up to 50 m²) via the web interface.

### Reconstruction — DISCOVERSE

**DISCOVERSE** (MIT, 2025) — Real2Sim framework. Internally: COLMAP (SfM) → 3D Gaussian Splatting → MJCF export. For the user — a single call:

```
Photos (15–30) → DISCOVERSE → MuJoCo scene (MJCF + collision geometry)
```

Photorealistic render at 650 FPS in MuJoCo.

Scale from photogrammetry is unknown (scale ambiguity). After reconstruction the user **marks two endpoints of a known object** (door, table, tile) in the Three.js 3D preview and enters the real size in meters.

```python
class ReferenceCalibration(BaseModel):
    """Calibration of reconstruction scale."""

    point_a: tuple[float, float, float]  # Point A in mesh coordinates
    point_b: tuple[float, float, float]  # Point B in mesh coordinates
    real_distance_m: float               # Real distance between A and B

async def reconstruct_and_calibrate(
    photos_dir: Path,
    calibration: ReferenceCalibration,
) -> SceneReconstruction:
    """Room reconstruction + scale calibration.

    Args:
        photos_dir: Directory with room photos.
        calibration: Reference measurement from user (two points + real size).

    Returns:
        Reconstructed scene at real scale.
    """
    scene = discoverse.real2sim(
        image_path=photos_dir,
        output_path=photos_dir / "mujoco_scene",
    )

    mesh_distance = np.linalg.norm(
        np.array(calibration.point_a) - np.array(calibration.point_b)
    )
    scale_factor = calibration.real_distance_m / mesh_distance
    scene.apply_scale(scale_factor)

    return SceneReconstruction(
        mesh_path=scene.mesh_path,
        mjcf_path=scene.mjcf_path,
        pointcloud_path=scene.pointcloud_path,
        dimensions=scene.get_dimensions(),
    )
```

### Recognition and structuring — Claude Vision

Claude Vision analyzes the photos **and** the mesh from DISCOVERSE, extracting structured data:
- Room dimensions (from the scaled DISCOVERSE mesh)
- Existing equipment, zones, doors, windows (from photos)

```python
async def analyze_scene(
    photos: list[Path],
    reconstruction: SceneReconstruction,
) -> SceneAnalysis:
    """Room analysis: photos + reconstruction → structured data.

    Args:
        photos: Room photos.
        reconstruction: DISCOVERSE result (mesh, MJCF).

    Returns:
        Dimensions, zones, existing equipment.
    """
    system_prompt = load_prompt("prompts/vision_analysis.md")

    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                *[image_content(photo) for photo in photos],
                text_content(f"Room mesh: {reconstruction.dimensions}"),
            ],
        }],
    )
    return SceneAnalysis.model_validate_json(response.content[0].text)
```

### Merge → SpaceModel

Reconstruction (DISCOVERSE MJCF) + analysis (Claude Vision) → `SpaceModel`:

```python
class Dimensions(BaseModel):
    """Room dimensions."""

    width_m: float
    length_m: float
    ceiling_m: float
    area_m2: float

class Zone(BaseModel):
    """Functional zone of the room."""

    name: str
    polygon: list[tuple[float, float]]  # 2D contour in meters
    area_m2: float

class Door(BaseModel):
    """Door in the room."""

    position: tuple[float, float]
    width_m: float

class Window(BaseModel):
    """Window in the room."""

    position: tuple[float, float]
    width_m: float

class ExistingEquipment(BaseModel):
    """Equipment already present in the room."""

    name: str
    category: str
    position: tuple[float, float, float]
    confidence: float

class SceneReconstruction(BaseModel):
    """DISCOVERSE Real2Sim result."""

    mesh_path: Path
    mjcf_path: Path            # Room scene MJCF (base for adding equipment)
    pointcloud_path: Path
    dimensions: Dimensions     # From scaled mesh (after reference calibration)

class SceneAnalysis(BaseModel):
    """Claude Vision analysis result."""

    zones: list[Zone]
    existing_equipment: list[ExistingEquipment]
    doors: list[Door]
    windows: list[Window]

class SpaceModel(BaseModel):
    """Room model for simulation."""

    dimensions: Dimensions
    zones: list[Zone]
    existing_equipment: list[ExistingEquipment]
    doors: list[Door]
    windows: list[Window]
    reconstruction: SceneReconstruction  # Reference to DISCOVERSE MJCF and mesh
```

### Web interface (stage 1)

1. Photo upload (drag-and-drop)
2. Wait for DISCOVERSE reconstruction (progress bar)
3. Three.js: 3D mesh preview. **Calibration**: user clicks two points on a known object (door, table) and enters real size in meters
4. 2D floor plan from above with recognized equipment and zones
5. Editing: adjust zones, equipment
6. “Confirm plan” button

---

## Knowledge-base — equipment catalog

Key architectural element from the original spec. Claude **does not invent** equipment — it only works from the catalog. Every `equipment_id` from Claude’s response is validated against the catalog; if not found — retry.

Only equipment that can be simulated in MuJoCo (has MJCF/URDF model):

```
knowledge-base/
└── equipment/
    ├── manipulators.json    # SO-101, xArm, Franka, UR5e, Koch...
    ├── conveyors.json       # Conveyor modules (with belt physics)
    ├── cameras.json         # Cameras (render in MuJoCo)
    └── fixtures.json        # Tables, shelves, stands, containers (static geometry)
```

Each equipment entry contains:

```python
class EquipmentEntry(BaseModel):
    """Equipment entry in the catalog."""

    id: str
    name: str
    type: Literal["manipulator", "conveyor", "camera", "fixture"]
    specs: dict                # reach, payload, dimensions, belt speed, etc.
    mjcf_source: MjcfSource    # menagerie_id or urdf_url
    price_usd: float | None = None
    purchase_url: str | None = None
    placement_rules: PlacementRules | None = None  # min zone, constraints

class MjcfSource(BaseModel):
    """MJCF model source."""

    menagerie_id: str | None = None    # e.g. "franka_emika_panda"
    robot_descriptions_id: str | None = None
    urdf_url: str | None = None        # Direct link to URDF
```

> Sensors, actuators, controllers — not in MVP (no MJCF models; needed only for real hardware).

The `type` field defines how equipment behaves in simulation:
- `manipulator` — controlled via IK/policy, performs pick/place/move
- `conveyor` — moves objects on belt, controlled via speed
- `camera` — renders image for inspection
- `fixture` — static geometry (tables, shelves), not controlled

---

## Module 2. AI recommendation

### Input

SpaceModel + user’s text description of the automation scenario.

Example: *“Dark kitchen, 3 workstations. Manipulator portions at station 2. Conveyor moves containers between stations. Need quality control with a camera.”*

### Process

Claude API receives:
- SpaceModel (JSON)
- Scenario text
- **Equipment catalog from knowledge-base**

```python
async def generate_recommendation(
    space: SpaceModel,
    scenario: str,
) -> Recommendation:
    """Generate robotization plan via Claude API.

    Args:
        space: Room model.
        scenario: User’s text description of the scenario.

    Returns:
        Robotization plan with equipment from the catalog.
    """
    catalog = load_equipment_catalog()
    system_prompt = load_prompt("prompts/recommendation.md")

    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": format_recommendation_context(
                space, scenario, catalog,
            ),
        }],
    )
    recommendation = parse_and_validate(response, catalog)
    return recommendation
```

**Validation**: every `equipment_id` is checked against the catalog. Prices come from the catalog, not from Claude’s response. On invalid id — retry (up to 2 times).

Claude returns **two formats**:

**1. Text plan** — human-readable description:
- Which equipment, why, where it is placed
- Sequence of actions
- Expected metrics

**2. Structured JSON** — machine-readable:

```python
class Recommendation(BaseModel):
    """Room automation plan."""

    equipment: list[EquipmentPlacement]
    work_objects: list[WorkObject]       # Objects for manipulation (parts, containers)
    target_positions: dict[str, tuple[float, float, float]]  # Name → coordinates mapping
    workflow_steps: list[WorkflowStep]
    expected_metrics: ExpectedMetrics

class EquipmentPlacement(BaseModel):
    """Placement of new equipment in the scene."""

    equipment_id: str      # ID from knowledge-base catalog
    position: tuple[float, float, float]
    orientation_deg: float
    purpose: str
    zone: str

class WorkObject(BaseModel):
    """Object for manipulation in simulation (part, box, container)."""

    name: str
    shape: Literal["box", "cylinder", "sphere"]
    size: tuple[float, float, float]  # For box: x,y,z. For cylinder: r,h,0
    mass_kg: float
    position: tuple[float, float, float]
    count: int = 1                     # Number of instances

class WorkflowStep(BaseModel):
    """Workflow step."""

    order: int
    action: str                    # "pick", "place", "move", "transport", "inspect", "wait"
    equipment_id: str | None = None  # None for "wait"
    target: str                    # Key from target_positions → 3D coordinates
    duration_s: float
    params: dict | None = None     # Extra params (speed for conveyor, etc.)
```

### Plan visualization

Frontend shows the recommendation:
- Three.js: 3D scene with room mesh + equipment outlines at positions
- Text plan on the side
- Buttons: “Confirm” / “Edit” (by text; Claude will revise)

---

## Module 3. Auto-download and scene assembly

### After plan confirmation

The system automatically:

1. **Downloads equipment models** from knowledge-base data:
```python
async def download_equipment_models(
    placements: list[EquipmentPlacement],
) -> dict[str, Path]:
    """Downloads MJCF/URDF models for all equipment from the recommendation.

    Args:
        placements: List of placements from Recommendation.

    Returns:
        Mapping equipment_id → model path.
    """
    catalog = load_equipment_catalog()
    models = {}
    for p in placements:
        entry = catalog[p.equipment_id]
        if entry.menagerie_id:
            models[p.equipment_id] = get_menagerie_model(entry.menagerie_id)
        elif entry.urdf_url:
            models[p.equipment_id] = await download_urdf(entry.urdf_url)
    return models
```

2. **Assembles the MJCF scene** — DISCOVERSE mesh as background + interactive objects:
```python
def generate_mjcf_scene(
    space: SpaceModel,
    recommendation: Recommendation,
    models: dict[str, Path],
    output_path: Path,
) -> Path:
    """Assembles final MJCF scene: room + equipment + objects.

    Args:
        space: Room model (DISCOVERSE MJCF + existing_equipment).
        recommendation: Automation plan.
        models: Mapping equipment_id → MJCF model path.
        output_path: Path to save the assembled scene.

    Returns:
        Path to the final MJCF file.
    """
    base_mjcf = load_mjcf(space.reconstruction.mjcf_path)

    # Existing equipment as separate interactive bodies
    # (simplified shapes on top of DISCOVERSE mesh)
    for eq in space.existing_equipment:
        add_static_body(base_mjcf, name=eq.name, pos=eq.position,
                        shape="box", size=estimate_size(eq.category))

    # New equipment from recommendation
    for placement in recommendation.equipment:
        model_path = models[placement.equipment_id]
        add_equipment_to_scene(base_mjcf, model_path,
                               pos=placement.position,
                               orientation=placement.orientation_deg)

    # Work objects for manipulation
    for obj in recommendation.work_objects:
        for i in range(obj.count):
            add_dynamic_body(base_mjcf, name=f"{obj.name}_{i}",
                             shape=obj.shape, size=obj.size,
                             mass=obj.mass_kg, pos=obj.position)

    save_mjcf(base_mjcf, output_path)
    return output_path
```

Three body types in the scene:
- **Background** — DISCOVERSE mesh (walls, floor, ceiling, visuals)
- **Static bodies** — existing_equipment (printers, tables), do not move but have collision
- **Dynamic bodies** — work_objects (parts, boxes), can be grasped and moved

3. **Installs dependencies** (if not already installed):
```bash
pip install mujoco mujoco-python-viewer lerobot robot_descriptions trimesh
```

---

## Module 4. Simulation and runs

### Launch

```python
async def run_simulation(
    scene_path: Path,
    workflow: list[WorkflowStep],
    catalog: dict[str, EquipmentEntry],
    target_positions: dict[str, tuple[float, float, float]],
    policy: LeRobotPolicy | None = None,  # None = scripted mode (MVP v1)
) -> SimResult:
    """Runs scene simulation in MuJoCo.

    Args:
        scene_path: Path to scene MJCF file.
        workflow: Workflow step sequence.
        catalog: Equipment catalog (for type resolution).
        target_positions: Mapping target name → 3D coordinates.
        policy: Trained policy (MVP v2), None for scripted.

    Returns:
        Simulation results with metrics.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    results = []
    for step in workflow:
        result = await execute_step(model, data, step, catalog, target_positions, policy)
        results.append(result)

    return SimResult(
        steps=results,
        metrics=compute_metrics(results),
    )
```

### Dispatch by action type

Each `WorkflowStep` is executed according to equipment type and action:

```python
async def execute_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    catalog: dict[str, EquipmentEntry],
    target_positions: dict[str, tuple[float, float, float]],
    policy: LeRobotPolicy | None,
) -> StepResult:
    """Executes one workflow step in simulation.

    Args:
        model: MuJoCo model.
        data: MuJoCo simulation data.
        step: Workflow step.
        catalog: Catalog for equipment type resolution.
        target_positions: Mapping target name → 3D coordinates.
        policy: Trained policy (optional).

    Returns:
        Step result: success, time, collisions.
    """
    if step.action == "wait":
        return await sim_wait(model, data, step.duration_s)

    equipment_type = catalog[step.equipment_id].type

    if equipment_type == "manipulator":
        if step.action in ("pick", "place", "move"):
            if policy:
                return await learned_manipulation(model, data, step, policy)
            return await scripted_manipulation(model, data, step, target_positions)

    elif equipment_type == "conveyor":
        if step.action == "transport":
            return await sim_conveyor(model, data, step)

    elif equipment_type == "camera":
        if step.action == "inspect":
            return await sim_camera_inspect(model, data, step, target_positions)

    raise ValueError(f"Unknown action '{step.action}' for {equipment_type}")
```

### Executors by equipment type

**Manipulator** — IK controller (scripted) or trained policy (learned):
```python
async def scripted_manipulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    target_positions: dict[str, tuple[float, float, float]],
) -> StepResult:
    """Manipulation via IK controller."""
    target_pos = target_positions[step.target]  # Resolve name → coordinates
    trajectory = compute_ik_trajectory(model, data, target_pos)
    return execute_trajectory(model, data, trajectory)

async def learned_manipulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    policy: LeRobotPolicy,
) -> StepResult:
    """Manipulation via trained policy (SmolVLA)."""
    obs = get_observation(model, data)
    while not is_done(model, data, step):
        action = policy.predict(obs)
        apply_action(model, data, action)
        mujoco.mj_step(model, data)
        obs = get_observation(model, data)
    return evaluate_result(model, data, step)
```

**Conveyor** — belt speed control:
```python
async def sim_conveyor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
) -> StepResult:
    """Conveyor simulation: moving objects on the belt."""
    conveyor_joint = find_joint(model, step.equipment_id)
    set_conveyor_speed(data, conveyor_joint, step.params.get("speed", 0.1))
    await sim_until(model, data, step.duration_s)
    return StepResult(success=True, duration_s=step.duration_s)
```

**Camera** — render + check target visibility:
```python
async def sim_camera_inspect(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    target_positions: dict[str, tuple[float, float, float]],
) -> StepResult:
    """Inspection simulation: check that camera sees the target."""
    image = render_camera(model, data, camera_name=step.equipment_id)
    target_pos = target_positions[step.target]
    visible = is_in_camera_fov(model, data, step.equipment_id, target_pos)
    return StepResult(
        success=visible,
        duration_s=0.1,
        image=image,
        error=None if visible else f"Target '{step.target}' not in camera FOV",
    )
```

### Simulation data models

```python
class StepResult(BaseModel):
    """Result of one simulation step."""

    success: bool
    duration_s: float
    collision_count: int = 0
    error: str | None = None
    image: np.ndarray | None = None  # For camera inspect

class SimResult(BaseModel):
    """Result of a full simulation run."""

    steps: list[StepResult]
    metrics: SimMetrics

class SimMetrics(BaseModel):
    """Simulation run metrics."""

    cycle_time_s: float
    success_rate: float       # 0.0–1.0
    collision_count: int
    failed_steps: list[int]   # Indices of failed steps
```

### Visualization

- **Built-in MuJoCo viewer** — interactive 3D run visualization
- **Web interface** — MuJoCo frame render → WebSocket → Three.js (for remote viewing)

---

## Module 5. Iterative improvement

### Loop

```
Run → Metrics → Claude analyzes → Corrections → New run
```

Up to 5 iterations. Claude receives:
- Current metrics (SimMetrics)
- Collision and error log
- Current scene configuration
- History of previous iterations

### What Claude corrects

1. **Robot positions** — closer/further to pick/place zone
2. **Object positions** — workspace optimization
3. **Trajectory parameters** — lift height, intermediate waypoints
4. **Add/remove objects** — e.g. add table or shelf if missing
5. **Robot model change** — if reach is insufficient, suggest another from catalog

```python
class PositionChange(BaseModel):
    """Equipment position change."""

    equipment_id: str
    new_position: tuple[float, float, float]
    new_orientation_deg: float | None = None

class EquipmentReplacement(BaseModel):
    """Replacement of equipment with another from the catalog."""

    old_equipment_id: str
    new_equipment_id: str   # Validated against catalog
    reason: str

class SceneCorrections(BaseModel):
    """Corrections from Claude after metrics analysis."""

    position_changes: list[PositionChange] | None = None
    add_equipment: list[EquipmentPlacement] | None = None
    remove_equipment: list[str] | None = None        # equipment_id
    replace_equipment: list[EquipmentReplacement] | None = None
    workflow_changes: list[WorkflowStep] | None = None  # Modified steps

class IterationLog(BaseModel):
    """Log of one iteration for Claude context."""

    iteration: int
    metrics: SimMetrics
    corrections_applied: SceneCorrections

async def iterate(
    scene_path: Path,
    metrics: SimMetrics,
    history: list[IterationLog],
    catalog: dict[str, EquipmentEntry],
) -> Path:
    """One improvement iteration via Claude.

    Args:
        scene_path: Path to current scene MJCF file.
        metrics: Metrics from the last run.
        history: Log of previous iterations.
        catalog: Equipment catalog (for validating replacements).

    Returns:
        Path to the corrected MJCF file.
    """
    # Read MJCF content — Claude cannot read files
    scene_xml = scene_path.read_text()

    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        system=load_prompt("prompts/iteration.md"),
        messages=[{
            "role": "user",
            "content": format_iteration_context(
                scene_xml=scene_xml,
                metrics=metrics,
                history=history,
                catalog=catalog,
            ),
        }],
    )
    corrections = SceneCorrections.model_validate_json(response.content[0].text)

    # If Claude suggested equipment change — download new models
    if corrections.replace_equipment:
        for replacement in corrections.replace_equipment:
            validate_equipment_id(replacement.new_equipment_id, catalog)
            await download_equipment_model(replacement.new_equipment_id)

    new_scene_path = apply_corrections(scene_path, corrections)
    return new_scene_path
```

### Stopping criteria

- `success_rate >= 0.95` and `collision_count == 0` → success
- 5 iterations with no improvement → stop, report to user
- User can stop manually

---

## Module 6. Policy training (MVP v2)

> This module is used **only if the recommendation includes manipulators**. For scenarios without robots (conveyors, cameras only) — the pipeline ends after Module 5.

### Pipeline: scripted → demonstrations → trained policy

**Step 1. Record demonstrations** — scripted controller (from Module 4, after successful Module 5 iterations) records trajectories in LeRobot dataset format:

```python
async def record_demonstrations(
    scene_path: Path,
    workflow: list[WorkflowStep],
    num_demos: int = 100,
    output_dir: Path = Path("data/projects/{id}/policies/demos"),
) -> Path:
    """Records successful scripted trajectories as demonstrations.

    Args:
        scene_path: Final MJCF scene (after iterations).
        workflow: Workflow steps.
        num_demos: Number of demonstrations (50–200).
        output_dir: Directory for LeRobot dataset.

    Returns:
        Path to the recorded dataset.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    dataset = LeRobotDataset(output_dir)

    for i in range(num_demos):
        data = mujoco.MjData(model)
        # Randomize initial object positions for diversity
        randomize_object_positions(data)

        obs_sequence = []
        for step in workflow:
            trajectory = compute_ik_trajectory(model, data, step)
            for action in trajectory:
                apply_action(model, data, action)
                mujoco.mj_step(model, data)
                obs_sequence.append(Observation(
                    image=render_camera(model, data),  # RGB from scene camera
                    joints=get_joint_positions(data),
                    action=action,
                ))

        dataset.add_episode(obs_sequence)

    dataset.save()
    return output_dir
```

**Step 2. Train SmolVLA** — fine-tune on recorded demonstrations:
```bash
python -m lerobot.scripts.train \
  --policy.type=smolvla \
  --dataset.repo_id=local:workspace_demos \
  --env.type=mujoco \
  --env.task=pick_and_place_custom
```

**Step 3. Evaluation** — run the trained policy in the same MuJoCo scene, compare metrics with scripted.

**Step 4. Iteration** — if metrics are below scripted, add more demonstrations (more randomization) and retrain.

### SmolVLA

- 450M parameters — runs on CPU/MacBook
- 30 Hz async inference
- Pre-trained on LeRobot community data
- Fine-tune on 50–200 demonstrations for the specific task
- **Observations**: RGB image from scene camera + joint positions

---

## MVP pipeline — end to end

```
┌─────────────────────────────────────┐
│  1. User                            │
│     • Uploads room photos           │
│     • Writes scenario in text       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  2. Capture (Module 1)               │
│     • DISCOVERSE → room MJCF        │
│     • Claude Vision → zones, equip.  │
│     • → SpaceModel                  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  3. Recommendation (Module 2)        │
│     • Claude API + catalog → JSON   │
│     • Visualization in Three.js     │
│     • User confirms                 │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  4. Assembly (Module 3)               │
│     • Download MJCF from catalog    │
│     • Room MJCF + robots            │
│     • → final MJCF scene            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  5. Simulation (Module 4)            │
│     • MuJoCo: scripted IK           │
│     • Metrics: time, success, coll.  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  6. Iterations (Module 5)       ◄──┐ │
│     • Claude corrects scene        │ │
│     • Download new models if       │ │
│     •   equipment changed          │ │
│     • Rerun ───────────────────────┘ │
│     • success_rate ≥ 0.95 → done     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  7. Training (Module 6, MVP v2)      │
│     • Only if manipulators present  │
│     • Record demonstrations (scripted) │
│     • LeRobot + SmolVLA fine-tune   │
│     • Evaluate learned vs scripted  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  8. Result                           │
│     • Final MJCF scene               │
│     • Trained policy (v2)            │
│     • Report with metrics           │
│     • Video of best run              │
└─────────────────────────────────────┘
```

---

## Project structure

```
lang2robo/
├── pyproject.toml
├── docker-compose.yml        # API + Web
├── .env.example
│
├── knowledge-base/
│   └── equipment/            # JSON equipment catalog (manipulators, conveyors, sensors...)
│
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entrypoint
│   │   ├── api/
│   │   │   ├── capture.py    # POST /capture — photo upload, analysis
│   │   │   ├── recommend.py  # POST /recommend — AI recommendation
│   │   │   ├── simulate.py   # POST /simulate — run simulation
│   │   │   └── iterate.py    # POST /iterate — improvement iteration
│   │   ├── models/           # Pydantic models
│   │   │   ├── space.py      # SpaceModel, Zone, Equipment
│   │   │   ├── recommendation.py  # Recommendation, RobotPlacement
│   │   │   └── simulation.py      # SimResult, SimMetrics
│   │   ├── services/
│   │   │   ├── vision.py     # Claude Vision photo analysis
│   │   │   ├── planner.py    # Claude recommendation + iterations
│   │   │   ├── scene.py      # MJCF scene generation
│   │   │   ├── simulator.py  # MuJoCo runs
│   │   │   └── downloader.py # Download models from Menagerie
│   │   └── core/
│   │       ├── config.py     # Settings (Pydantic)
│   │       ├── claude.py      # Claude API client
│   │       └── prompts.py     # load_prompt() — load prompts from prompts/
│   └── tests/
│       ├── test_vision.py
│       ├── test_planner.py
│       ├── test_scene.py
│       └── test_simulator.py
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── PhotoUpload.tsx
│   │   │   ├── FloorPlanEditor.tsx
│   │   │   ├── SceneViewer3D.tsx    # Three.js MuJoCo scene
│   │   │   ├── RecommendationView.tsx
│   │   │   ├── SimulationPlayer.tsx
│   │   │   └── MetricsDashboard.tsx
│   │   └── types/
│   │       └── index.ts
│   └── tsconfig.json
│
├── prompts/
│   ├── vision_analysis.md     # System prompt: room photo analysis
│   ├── recommendation.md      # System prompt: robotization plan generation
│   └── iteration.md           # System prompt: scene correction from metrics
│
├── models/                    # Cache of downloaded MJCF models
│
├── data/
│   └── projects/
│       └── {project_id}/
│           ├── photos/            # User’s source photos
│           ├── reconstruction/   # DISCOVERSE output (mesh, point cloud, MJCF)
│           ├── recommendation/   # Claude JSON recommendation
│           ├── scenes/           # MJCF scenes per iteration (v1.xml, v2.xml...)
│           ├── simulations/      # Run metrics + videos
│           ├── policies/         # Trained policies + demonstrations (MVP v2)
│           └── report.json        # Final report
│
└── scripts/
    ├── train_policy.py        # LeRobot fine-tune (MVP v2)
    └── record_demos.py        # Record demonstrations from scripted (MVP v2)
```

---

## Dependencies

### Backend (Python)

```toml
[project]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]",
    "pydantic>=2.0",
    "anthropic>=0.40",
    "mujoco>=3.0",
    "robot-descriptions",
    "discoverse",
    "trimesh",
    "numpy",
    "pillow",
]

[project.optional-dependencies]
training = [
    "lerobot",
    "torch",
]
```

### Frontend (TypeScript)

```json
{
  "dependencies": {
    "react": "^19",
    "three": "^0.170",
    "@react-three/fiber": "^9",
    "@react-three/drei": "^10"
  }
}
```

---

## Environment variables

```env
ANTHROPIC_API_KEY=sk-ant-...   # Only external service
```

---

## Run (Demo MVP)

```bash
git clone https://github.com/user/lang2robo
cd lang2robo
pip install -e ".[training]"
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app --reload
# Open http://localhost:8000
```

---

## Example scenarios

### Scenario 1: 3D printing studio (robot + conveyor + camera)

**Space**: 30 m², 5 Bambu Lab printers, work table, shelf.

**User scenario**: *“Robot removes finished prints from printer build plates, places them on the conveyor. Conveyor moves them to the post-processing table. Camera detects print failures.”*

**Claude recommendation**:
```
equipment: [franka_emika_panda, conveyor_500mm, camera_overhead]

work_objects: [
  WorkObject("finished_print", shape="box", size=(0.05, 0.05, 0.04), mass_kg=0.1,
             position=(1.0, 1.0, 0.85), count=5)
]

target_positions: {
  "printer_1_bed": (1.0, 1.0, 0.85),
  "printer_2_bed": (2.5, 1.0, 0.85),
  "printer_3_bed": (4.0, 1.0, 0.85),
  "conveyor_start": (3.0, 2.0, 0.85),
  "conveyor_end":   (3.0, 4.0, 0.85),
  "post_table":     (3.0, 4.5, 0.85),
}

workflow_steps: [
  (1, "inspect",   camera_overhead,    "printer_1_bed"),
  (2, "pick",      franka_emika_panda, "printer_1_bed"),
  (3, "place",     franka_emika_panda, "conveyor_start"),
  (4, "transport", conveyor_500mm,     "conveyor_end", params={"speed": 0.05}),
  (5, "wait",      None,               "next_print_ready"),
]
```

**MJCF scene**: DISCOVERSE background + 5 printers (static bodies) + Franka + conveyor + camera + 5 parts (dynamic bodies).

**Simulation**: inspect → pick → place → transport → wait. All steps dispatched by equipment type.

**Iteration**: Franka could not reach printer_3 → Claude shifts robot to (2.5, 1.5, 0.0) → rerun → success.

**Training (MVP v2)**: Manipulator present → 100 demonstrations with randomized part positions → SmolVLA fine-tune.

---

### Scenario 2: Pickup point — order fulfillment (NO robot)

**Space**: 20 m², shelf, pickup window, reception table.

**User scenario**: *“Pickup point. Conveyor moves parcels from reception table to the shelf. Camera scans barcodes for sorting. No robot.”*

**Claude recommendation**:
```
equipment: [conveyor_1000mm, camera_barcode]

work_objects: [
  WorkObject("parcel_small", shape="box", size=(0.20, 0.15, 0.10), mass_kg=0.5,
             position=(1.0, 1.0, 0.85), count=10),
  WorkObject("parcel_large", shape="box", size=(0.40, 0.30, 0.20), mass_kg=2.0,
             position=(1.0, 1.5, 0.85), count=5),
]

target_positions: {
  "reception_table": (1.0, 1.0, 0.85),
  "conveyor_start":  (1.5, 1.0, 0.85),
  "conveyor_end":    (4.0, 1.0, 0.85),
  "shelf_zone":      (4.5, 1.0, 0.85),
}

workflow_steps: [
  (1, "inspect",   camera_barcode,  "reception_table"),
  (2, "transport", conveyor_1000mm, "conveyor_end", params={"speed": 0.08}),
  (3, "wait",      None,            "next_parcel"),
]
```

**MJCF scene**: DISCOVERSE background + shelf/table (static bodies) + conveyor + camera + 15 parcels in two sizes (dynamic bodies).

**Simulation**: inspect → transport → wait. No pick/place — no IK. Conveyor moves parcels via MuJoCo friction.

**Iteration**: Camera could not see reception_table (bad angle). Claude adjusts camera position → rerun → visible=true.

**Training**: No manipulators → **Module 6 is skipped**.

---

### Scenario 3: Electronics repair workshop (robot + camera, no conveyor)

**Space**: 15 m², soldering station, microscope, table with components.

**User scenario**: *“Repair workshop. Robot picks a board from the intake table, brings it to the microscope camera for inspection, then moves it to the soldering station.”*

**Claude recommendation**:
```
equipment: [koch_v1_1, camera_microscope]

work_objects: [
  WorkObject("pcb_board", shape="box", size=(0.10, 0.07, 0.002), mass_kg=0.05,
             position=(0.5, 0.5, 0.75), count=3)
]

target_positions: {
  "intake_table":      (0.5, 0.5, 0.75),
  "microscope_fov":    (1.5, 0.5, 0.80),
  "soldering_station": (2.5, 0.5, 0.75),
}

workflow_steps: [
  (1, "pick",    koch_v1_1,         "intake_table"),
  (2, "move",    koch_v1_1,         "microscope_fov"),
  (3, "inspect", camera_microscope, "microscope_fov"),
  (4, "place",   koch_v1_1,         "soldering_station"),
]
```

**MJCF scene**: DISCOVERSE background + soldering station/microscope (static bodies) + Koch v1.1 + camera + 3 boards (dynamic bodies, mass=0.05 kg).

**Simulation**: pick → move (holding board) → inspect → place. Koch — small arm for precise operations.

**Iteration**: Koch could not reach soldering_station (reach 0.28 m, need 0.35 m). Claude suggests `replace_equipment`: Koch → Franka. System downloads Franka from Menagerie, reassembles scene → rerun → success.

**Training (MVP v2)**: Manipulator present → 100 demonstrations with randomized board positions → SmolVLA fine-tune.

---

### Scenario 4: Dark kitchen (robot + conveyor + camera)

**Space**: 35 m², stove, refrigerator, 3 workstations.

**User scenario**: *“Dark kitchen. Manipulator portions at station 2. Conveyor feeds containers from station 1 to station 3. Camera monitors portioning.”*

**Claude recommendation**:
```
equipment: [franka_emika_panda, conveyor_500mm, camera_overhead]

work_objects: [
  WorkObject("food_container", shape="box", size=(0.15, 0.10, 0.08), mass_kg=0.3,
             position=(1.0, 1.5, 0.85), count=5)
]

target_positions: {
  "station_1":       (1.0, 1.5, 0.85),
  "station_2":       (3.0, 1.5, 0.85),
  "station_3":       (5.0, 1.5, 0.85),
  "conveyor_start":  (1.0, 2.0, 0.85),
  "conveyor_end":    (5.0, 2.0, 0.85),
}

workflow_steps: [
  (1, "pick",      franka_emika_panda, "station_1"),
  (2, "place",     franka_emika_panda, "station_2"),
  (3, "wait",      None,               "portioning_done"),
  (4, "inspect",   camera_overhead,    "station_2"),
  (5, "pick",      franka_emika_panda, "station_2"),
  (6, "place",     franka_emika_panda, "conveyor_start"),
  (7, "transport", conveyor_500mm,     "conveyor_end", params={"speed": 0.05}),
]
```

**MJCF scene**: DISCOVERSE background + stove/refrigerator/tables (static bodies) + Franka + conveyor + camera + 5 containers (dynamic bodies).

**Simulation**: pick → place → wait → inspect → pick → place → transport. Full cycle with manipulation and conveyor.

**Iteration**: Arm ↔ table collision at step 6. Claude shifts Franka higher (z += 0.1) → rerun → success.

**Training (MVP v2)**: Manipulator present → 100 demonstrations → SmolVLA fine-tune.
