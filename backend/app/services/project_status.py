"""Project status persistence — CRUD for status.json per project."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from backend.app.core.config import get_settings
from backend.app.models.project import (
    PhaseRecord,
    PipelinePhase,
    ProjectDetail,
    ProjectStatus,
)

__all__ = [
    "get_project_dir",
    "create_project_status",
    "load_project_status",
    "advance_phase",
    "list_all_projects",
    "load_project_detail",
]


def get_project_dir(project_id: str) -> Path:
    """Get project data directory path.

    Args:
        project_id: Project identifier.

    Returns:
        Path to project directory.
    """
    return get_settings().DATA_DIR / "projects" / project_id


def _status_path(project_id: str) -> Path:
    """Return path to a project's status.json."""
    return get_project_dir(project_id) / "status.json"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


def create_project_status(project_id: str, name: str = "") -> ProjectStatus:
    """Create initial status.json for a new project.

    Args:
        project_id: Project identifier.
        name: Human-readable project name.

    Returns:
        Newly created project status.
    """
    now = _now()
    status = ProjectStatus(
        id=project_id,
        name=name,
        current_phase="upload",
        created_at=now,
        updated_at=now,
        phases_completed=[PhaseRecord(phase="upload", completed_at=now)],
    )
    _write_status(status)
    return status


def load_project_status(project_id: str) -> ProjectStatus:
    """Read status.json for an existing project.

    Args:
        project_id: Project identifier.

    Returns:
        Project status.

    Raises:
        HTTPException: If project not found (404).
    """
    path = _status_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return ProjectStatus.model_validate_json(path.read_text(encoding="utf-8"))


def advance_phase(project_id: str, phase: PipelinePhase) -> ProjectStatus:
    """Mark a pipeline phase as completed and update status.json.

    Args:
        project_id: Project identifier.
        phase: Phase that just completed.

    Returns:
        Updated project status.
    """
    status = load_project_status(project_id)
    now = _now()
    status.current_phase = phase
    status.updated_at = now
    status.phases_completed.append(PhaseRecord(phase=phase, completed_at=now))
    _write_status(status)
    return status


def list_all_projects() -> list[ProjectStatus]:
    """Scan all project directories and return their statuses.

    Returns:
        List of project statuses sorted by updated_at descending.
    """
    projects_root = get_settings().DATA_DIR / "projects"
    if not projects_root.exists():
        return []
    statuses = _collect_statuses(projects_root)
    return sorted(statuses, key=lambda s: s.updated_at, reverse=True)


def _collect_statuses(projects_root: Path) -> list[ProjectStatus]:
    """Read status.json from each project subdirectory.

    Args:
        projects_root: Root directory containing project folders.

    Returns:
        List of successfully loaded project statuses.
    """
    statuses: list[ProjectStatus] = []
    for status_file in projects_root.glob("*/status.json"):
        try:
            status = ProjectStatus.model_validate_json(
                status_file.read_text(encoding="utf-8")
            )
            statuses.append(status)
        except (json.JSONDecodeError, ValueError):
            continue
    return statuses


def load_project_detail(project_id: str) -> ProjectDetail:
    """Load full project data for state restoration.

    Args:
        project_id: Project identifier.

    Returns:
        Project status with all available phase data.

    Raises:
        HTTPException: If project not found (404).
    """
    status = load_project_status(project_id)
    project_dir = get_project_dir(project_id)
    return ProjectDetail(
        status=status,
        dimensions=_load_dimensions(project_dir),
        recommendation=_load_recommendation(project_dir),
        sim_result=_load_sim_result(project_dir),
        iteration_history=_load_iteration_history(project_dir),
    )


def _load_dimensions(project_dir: Path):
    """Load dimensions from reconstruction metadata if available."""
    from backend.app.models.space import Dimensions

    meta_path = project_dir / "reconstruction_meta.json"
    if not meta_path.exists():
        return None
    import json as _json

    data = _json.loads(meta_path.read_text(encoding="utf-8"))
    dims = data.get("dimensions")
    return Dimensions.model_validate(dims) if dims else None


def _load_recommendation(project_dir: Path):
    """Load recommendation.json if available."""
    from backend.app.models.recommendation import Recommendation

    path = project_dir / "recommendation" / "recommendation.json"
    if not path.exists():
        return None
    return Recommendation.model_validate_json(path.read_text(encoding="utf-8"))


def _load_sim_result(project_dir: Path):
    """Load latest simulation result if available."""
    from backend.app.models.simulation import SimResult

    path = project_dir / "simulations" / "latest.json"
    if not path.exists():
        return None
    return SimResult.model_validate_json(path.read_text(encoding="utf-8"))


def _load_iteration_history(project_dir: Path) -> list:
    """Load iteration history if available."""
    from backend.app.models.iteration import IterationLog

    path = project_dir / "simulations" / "iteration_history.json"
    if not path.exists():
        return []
    import json as _json

    raw = _json.loads(path.read_text(encoding="utf-8"))
    return [IterationLog.model_validate(item) for item in raw]


def _write_status(status: ProjectStatus) -> None:
    """Write project status to disk.

    Args:
        status: Project status to persist.
    """
    path = _status_path(status.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(status.model_dump_json(indent=2), encoding="utf-8")
