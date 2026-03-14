"""Iteration API — Claude-driven scene optimization loop."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.claude import get_claude_client
from backend.app.models.iteration import IterationLog
from backend.app.models.recommendation import Recommendation
from backend.app.models.simulation import SimResult
from backend.app.services.catalog import load_equipment_catalog
from backend.app.services.iteration import run_iteration_loop
from backend.app.services.project_status import advance_phase, get_project_dir

__all__ = ["router"]

router = APIRouter(prefix="/api/projects", tags=["iterate"])


class IterateRequest(BaseModel):
    """Request body for iteration loop.

    Args:
        max_iterations: Maximum number of improvement iterations.
    """

    max_iterations: int = Field(default=5, ge=1, le=10)


class IterateResponse(BaseModel):
    """Response from iteration loop.

    Args:
        result: Final simulation result.
        history: Iteration log entries.
        iterations_run: Number of iterations actually executed.
        converged: Whether success criteria were met.
    """

    result: SimResult
    history: list[IterationLog]
    iterations_run: int
    converged: bool


@router.post("/{project_id}/iterate", response_model=IterateResponse)
async def iterate(
    project_id: str,
    request: IterateRequest | None = None,
) -> IterateResponse:
    """Run improvement iteration loop on project.

    Args:
        project_id: Project identifier.
        request: Iteration parameters.

    Returns:
        Final result and iteration history.
    """
    max_iter = request.max_iterations if request else 5
    recommendation = _load_recommendation(project_id)
    catalog = load_equipment_catalog()
    scene_path = _find_latest_scene(project_id)
    client = get_claude_client()

    result, history = await run_iteration_loop(
        scene_path,
        recommendation,
        catalog,
        client,
        max_iter,
    )

    _save_iteration_results(project_id, result, history)
    advance_phase(project_id, "iterate")

    converged = result.metrics.success_rate >= 0.95 and result.metrics.collision_count == 0
    return IterateResponse(
        result=result,
        history=history,
        iterations_run=len(history),
        converged=converged,
    )


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
        raise HTTPException(404, f"No scene files for {project_id}")
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


def _save_iteration_results(
    project_id: str,
    result: SimResult,
    history: list[IterationLog],
) -> None:
    """Save iteration results to project directory.

    Args:
        project_id: Project identifier.
        result: Final simulation result.
        history: Iteration history.
    """
    sim_dir = get_project_dir(project_id) / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    (sim_dir / "final_result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    history_data = [h.model_dump() for h in history]
    (sim_dir / "iteration_history.json").write_text(
        json.dumps(history_data, indent=2),
        encoding="utf-8",
    )
