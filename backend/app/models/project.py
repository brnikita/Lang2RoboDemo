"""Project status models for pipeline phase tracking."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.models.iteration import IterationLog
from backend.app.models.recommendation import Recommendation
from backend.app.models.simulation import SimResult
from backend.app.models.space import Dimensions

__all__ = [
    "PipelinePhase",
    "PhaseRecord",
    "ProjectStatus",
    "ProjectDetail",
]

PipelinePhase = Literal["upload", "calibrate", "recommend", "build-scene", "simulate", "iterate"]


class PhaseRecord(BaseModel):
    """Record of a completed pipeline phase.

    Args:
        phase: Pipeline phase identifier.
        completed_at: Timestamp when the phase completed.
    """

    phase: PipelinePhase
    completed_at: datetime


class ProjectStatus(BaseModel):
    """Persistent project status stored in status.json.

    Args:
        id: Project UUID.
        name: Human-readable project name.
        current_phase: Most recently completed pipeline phase.
        created_at: Project creation timestamp.
        updated_at: Last status update timestamp.
        phases_completed: Ordered list of completed phase records.
    """

    id: str
    name: str = ""
    current_phase: PipelinePhase
    created_at: datetime
    updated_at: datetime
    phases_completed: list[PhaseRecord] = Field(default_factory=list)


class ProjectDetail(BaseModel):
    """Full project data for state restoration.

    Args:
        status: Project status metadata.
        dimensions: Room dimensions from reconstruction.
        recommendation: Generated automation plan.
        sim_result: Latest simulation result.
        iteration_history: Log of improvement iterations.
    """

    status: ProjectStatus
    dimensions: Dimensions | None = None
    recommendation: Recommendation | None = None
    sim_result: SimResult | None = None
    iteration_history: list[IterationLog] = Field(default_factory=list)
