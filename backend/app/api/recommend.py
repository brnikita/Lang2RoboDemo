"""Recommendation API — AI-generated automation plans."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.claude import get_claude_client
from backend.app.core.config import get_settings
from backend.app.models.recommendation import Recommendation
from backend.app.models.space import SpaceModel
from backend.app.services.planner import generate_recommendation

__all__ = ["router"]

router = APIRouter(prefix="/api/recommend", tags=["recommend"])


class RecommendRequest(BaseModel):
    """Request body for recommendation generation.

    Args:
        project_id: Project identifier.
        scenario: User's text description of desired automation.
    """

    project_id: str
    scenario: str


@router.post("", response_model=Recommendation)
async def create_recommendation(
    request: RecommendRequest,
) -> Recommendation:
    """Generate an automation plan for the project.

    Args:
        request: Project ID and scenario text.

    Returns:
        Validated recommendation with equipment placements and workflow.
    """
    space_model = _load_space_model(request.project_id)
    client = get_claude_client()

    recommendation = await generate_recommendation(
        client, space_model, request.scenario,
    )

    _save_recommendation(request.project_id, recommendation)
    return recommendation


def _load_space_model(project_id: str) -> SpaceModel:
    """Load SpaceModel from project directory.

    Args:
        project_id: Project identifier.

    Returns:
        SpaceModel instance.

    Raises:
        HTTPException: If SpaceModel not found.
    """
    settings = get_settings()
    path = settings.DATA_DIR / "projects" / project_id / "space_model.json"
    if not path.exists():
        raise HTTPException(
            404,
            f"SpaceModel not found for project {project_id}. "
            f"Run capture and calibration first.",
        )
    return SpaceModel.model_validate_json(path.read_text(encoding="utf-8"))


def _save_recommendation(
    project_id: str, recommendation: Recommendation,
) -> None:
    """Save recommendation to project directory.

    Args:
        project_id: Project identifier.
        recommendation: Recommendation to save.
    """
    settings = get_settings()
    rec_dir = settings.DATA_DIR / "projects" / project_id / "recommendation"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_path = rec_dir / "recommendation.json"
    rec_path.write_text(
        recommendation.model_dump_json(indent=2), encoding="utf-8",
    )
