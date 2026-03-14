"""Iteration and scene correction models."""

from pydantic import BaseModel

from backend.app.models.recommendation import EquipmentPlacement, WorkflowStep
from backend.app.models.simulation import SimMetrics

__all__ = [
    "EquipmentReplacement",
    "IterationLog",
    "PositionChange",
    "SceneCorrections",
]


class PositionChange(BaseModel):
    """Change to an equipment's position in the scene.

    Args:
        equipment_id: ID of the equipment to move.
        new_position: New 3D position (x, y, z) in meters.
        new_orientation_deg: New rotation around Z-axis in degrees.
    """

    equipment_id: str
    new_position: tuple[float, float, float]
    new_orientation_deg: float | None = None


class EquipmentReplacement(BaseModel):
    """Replacement of one equipment with another from the catalog.

    Args:
        old_equipment_id: ID of the equipment to replace.
        new_equipment_id: ID of the replacement (validated against catalog).
        reason: Why this replacement is needed.
    """

    old_equipment_id: str
    new_equipment_id: str
    reason: str


class SceneCorrections(BaseModel):
    """Corrections proposed by Claude after analyzing simulation metrics.

    Args:
        position_changes: Equipment position adjustments.
        add_equipment: New equipment to add.
        remove_equipment: Equipment IDs to remove.
        replace_equipment: Equipment replacements.
        workflow_changes: Modified workflow steps.
    """

    position_changes: list[PositionChange] | None = None
    add_equipment: list[EquipmentPlacement] | None = None
    remove_equipment: list[str] | None = None
    replace_equipment: list[EquipmentReplacement] | None = None
    workflow_changes: list[WorkflowStep] | None = None


class IterationLog(BaseModel):
    """Log entry for one iteration of the improvement loop.

    Args:
        iteration: Iteration number (1-based).
        metrics: Simulation metrics from this iteration.
        corrections_applied: Corrections that were applied before this run.
    """

    iteration: int
    metrics: SimMetrics
    corrections_applied: SceneCorrections
