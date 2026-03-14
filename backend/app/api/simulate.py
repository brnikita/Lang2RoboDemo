"""Simulation API — scene building and MuJoCo runs."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.models.recommendation import Recommendation
from backend.app.models.simulation import SimResult
from backend.app.models.space import SpaceModel
from backend.app.services.catalog import load_equipment_catalog
from backend.app.services.downloader import download_equipment_models
from backend.app.services.project_status import advance_phase, get_project_dir
from backend.app.services.scene import generate_mjcf_scene, validate_mjcf
from backend.app.services.simulator import run_simulation, run_visual_simulation

__all__ = ["router"]

router = APIRouter(prefix="/api/projects", tags=["simulate"])


class BuildSceneResponse(BaseModel):
    """Response for scene build endpoint.

    Args:
        scene_path: Path to generated MJCF file.
        valid: Whether MuJoCo can load the scene.
        equipment_count: Number of equipment bodies.
        work_object_count: Number of work object bodies.
    """

    scene_path: str
    valid: bool
    equipment_count: int
    work_object_count: int


@router.post("/{project_id}/build-scene", response_model=BuildSceneResponse)
async def build_scene(project_id: str) -> BuildSceneResponse:
    """Download models and build MJCF scene from recommendation.

    Args:
        project_id: Project identifier.

    Returns:
        Scene metadata and validation status.
    """
    space = _load_space_model(project_id)
    recommendation = _load_recommendation(project_id)
    catalog = load_equipment_catalog()

    equipment_ids = [p.equipment_id for p in recommendation.equipment]
    model_dirs = await download_equipment_models(equipment_ids)

    scenes_dir = get_project_dir(project_id) / "scenes"
    scene_path = scenes_dir / "v1.xml"

    generate_mjcf_scene(
        space,
        recommendation,
        model_dirs,
        catalog,
        scene_path,
    )

    valid = validate_mjcf(scene_path)
    total_objects = sum(obj.count for obj in recommendation.work_objects)

    advance_phase(project_id, "build-scene")

    return BuildSceneResponse(
        scene_path=str(scene_path),
        valid=valid,
        equipment_count=len(recommendation.equipment),
        work_object_count=total_objects,
    )


@router.post("/{project_id}/view")
async def launch_viewer(project_id: str) -> dict:
    """Launch MuJoCo interactive viewer with workflow playback.

    Opens the viewer in a background thread and runs the full
    workflow simulation with real-time visualization.

    Args:
        project_id: Project identifier.

    Returns:
        Status message with scene path.
    """
    recommendation = _load_recommendation(project_id)
    catalog = load_equipment_catalog()
    scene_path = _find_latest_scene(project_id)

    import threading

    thread = threading.Thread(
        target=_run_visual_in_thread,
        args=(scene_path, recommendation, catalog),
        daemon=True,
    )
    thread.start()
    return {"status": "viewer_launched", "scene": str(scene_path)}


def _run_visual_in_thread(
    scene_path: Path,
    recommendation: "Recommendation",
    catalog: dict,
) -> None:
    """Run visual simulation in a dedicated thread with its own event loop.

    Args:
        scene_path: Path to MJCF scene file.
        recommendation: Project recommendation with workflow.
        catalog: Equipment catalog keyed by ID.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            run_visual_simulation(
                scene_path,
                recommendation.workflow_steps,
                catalog,
                recommendation.target_positions,
            ),
        )
    finally:
        loop.close()


def _load_space_model(project_id: str) -> SpaceModel:
    """Load SpaceModel from project.

    Args:
        project_id: Project identifier.

    Returns:
        SpaceModel instance.

    Raises:
        HTTPException: If not found.
    """
    path = get_project_dir(project_id) / "space_model.json"
    if not path.exists():
        raise HTTPException(404, f"SpaceModel not found for {project_id}")
    return SpaceModel.model_validate_json(path.read_text(encoding="utf-8"))


@router.post("/{project_id}/simulate", response_model=SimResult)
async def simulate(project_id: str) -> SimResult:
    """Run simulation on the latest scene.

    Args:
        project_id: Project identifier.

    Returns:
        Simulation result with per-step outcomes and metrics.
    """
    recommendation = _load_recommendation(project_id)
    catalog = load_equipment_catalog()
    scene_path = _find_latest_scene(project_id)

    result = await run_simulation(
        scene_path,
        recommendation.workflow_steps,
        catalog,
        recommendation.target_positions,
    )

    sim_dir = get_project_dir(project_id) / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "latest.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    advance_phase(project_id, "simulate")
    return result


def _find_latest_scene(project_id: str) -> Path:
    """Find the latest scene MJCF file.

    Args:
        project_id: Project identifier.

    Returns:
        Path to latest scene file.

    Raises:
        HTTPException: If no scene found.
    """
    scenes_dir = get_project_dir(project_id) / "scenes"
    if not scenes_dir.exists():
        raise HTTPException(404, f"No scenes for {project_id}")
    xmls = sorted(scenes_dir.glob("v*.xml"))
    if not xmls:
        raise HTTPException(404, f"No scene files in {scenes_dir}")
    return xmls[-1]


def _load_recommendation(project_id: str) -> Recommendation:
    """Load recommendation from project.

    Args:
        project_id: Project identifier.

    Returns:
        Recommendation instance.

    Raises:
        HTTPException: If not found.
    """
    path = get_project_dir(project_id) / "recommendation" / "recommendation.json"
    if not path.exists():
        raise HTTPException(404, f"Recommendation not found for {project_id}")
    return Recommendation.model_validate_json(
        path.read_text(encoding="utf-8"),
    )
