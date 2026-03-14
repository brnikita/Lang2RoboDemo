"""Projects API — list and retrieve project status."""

from fastapi import APIRouter

from backend.app.models.project import ProjectDetail, ProjectStatus
from backend.app.services.project_status import list_all_projects, load_project_detail

__all__ = ["router"]

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectStatus])
async def get_projects() -> list[ProjectStatus]:
    """List all projects sorted by last update.

    Returns:
        List of project statuses.
    """
    return list_all_projects()


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str) -> ProjectDetail:
    """Get full project detail for state restoration.

    Args:
        project_id: Project identifier.

    Returns:
        Project status with all available phase data.
    """
    return load_project_detail(project_id)
