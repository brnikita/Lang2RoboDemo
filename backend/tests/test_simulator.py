"""Tests for MuJoCo simulation runner."""

from backend.app.models.simulation import StepResult
from backend.app.services.simulator import compute_metrics


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_all_success(self) -> None:
        results = [
            StepResult(success=True, duration_s=2.0),
            StepResult(success=True, duration_s=3.0),
        ]
        metrics = compute_metrics(results)
        assert metrics.cycle_time_s == 5.0
        assert metrics.success_rate == 1.0
        assert metrics.failed_steps == []

    def test_partial_failure(self) -> None:
        results = [
            StepResult(success=True, duration_s=2.0),
            StepResult(success=False, duration_s=0.0, error="IK failed"),
            StepResult(success=True, duration_s=3.0),
        ]
        metrics = compute_metrics(results)
        assert abs(metrics.success_rate - 2 / 3) < 1e-6
        assert metrics.failed_steps == [1]

    def test_empty_results(self) -> None:
        metrics = compute_metrics([])
        assert metrics.cycle_time_s == 0.0
        assert metrics.success_rate == 0.0

    def test_collision_counting(self) -> None:
        results = [
            StepResult(success=True, duration_s=2.0, collision_count=5),
            StepResult(success=True, duration_s=3.0, collision_count=3),
        ]
        metrics = compute_metrics(results)
        assert metrics.collision_count == 8

    def test_all_failure(self) -> None:
        results = [
            StepResult(success=False, duration_s=0, error="err1"),
            StepResult(success=False, duration_s=0, error="err2"),
        ]
        metrics = compute_metrics(results)
        assert metrics.success_rate == 0.0
        assert metrics.failed_steps == [0, 1]
