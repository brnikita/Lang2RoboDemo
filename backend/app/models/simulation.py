"""Simulation result models."""

from pydantic import BaseModel, Field

__all__ = [
    "SimMetrics",
    "SimResult",
    "StepResult",
]


class StepResult(BaseModel):
    """Result of a single simulation step.

    Args:
        success: Whether the step completed successfully.
        duration_s: Actual step duration in seconds.
        collision_count: Number of collisions during this step.
        error: Error description if the step failed.
    """

    success: bool
    duration_s: float = Field(ge=0)
    collision_count: int = Field(default=0, ge=0)
    error: str | None = None


class SimMetrics(BaseModel):
    """Aggregate metrics for a full simulation run.

    Args:
        cycle_time_s: Total cycle time in seconds.
        success_rate: Fraction of successful steps (0.0 to 1.0).
        collision_count: Total collisions across all steps.
        failed_steps: Indices of failed steps (0-based).
    """

    cycle_time_s: float = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    collision_count: int = Field(default=0, ge=0)
    failed_steps: list[int] = Field(default_factory=list)


class SimResult(BaseModel):
    """Complete result of a simulation run.

    Args:
        steps: Results for each workflow step.
        metrics: Aggregate simulation metrics.
    """

    steps: list[StepResult]
    metrics: SimMetrics
