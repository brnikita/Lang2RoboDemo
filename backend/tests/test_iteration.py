"""Tests for iteration module."""

import xml.etree.ElementTree as ET
from pathlib import Path

from backend.app.models.iteration import PositionChange
from backend.app.models.simulation import SimMetrics
from backend.app.services.iteration import (
    _apply_position_change,
    _is_converged,
    _next_scene_path,
    _remove_body,
)


class TestConvergenceCheck:
    """Tests for convergence criteria."""

    def test_converged(self) -> None:
        metrics = SimMetrics(
            cycle_time_s=10.0,
            success_rate=0.98,
            collision_count=0,
        )
        assert _is_converged(metrics)

    def test_not_converged_low_success(self) -> None:
        metrics = SimMetrics(
            cycle_time_s=10.0,
            success_rate=0.80,
            collision_count=0,
        )
        assert not _is_converged(metrics)

    def test_not_converged_collisions(self) -> None:
        metrics = SimMetrics(
            cycle_time_s=10.0,
            success_rate=0.98,
            collision_count=5,
        )
        assert not _is_converged(metrics)

    def test_exact_threshold(self) -> None:
        metrics = SimMetrics(
            cycle_time_s=10.0,
            success_rate=0.95,
            collision_count=0,
        )
        assert _is_converged(metrics)


class TestNextScenePath:
    """Tests for scene version path generation."""

    def test_v1_to_v2(self) -> None:
        result = _next_scene_path(Path("/scenes/v1.xml"))
        assert result == Path("/scenes/v2.xml")

    def test_v5_to_v6(self) -> None:
        result = _next_scene_path(Path("/scenes/v5.xml"))
        assert result == Path("/scenes/v6.xml")

    def test_non_versioned(self) -> None:
        result = _next_scene_path(Path("/scenes/scene.xml"))
        assert result == Path("/scenes/v2.xml")


class TestApplyCorrections:
    """Tests for MJCF corrections application."""

    def _make_scene_xml(self) -> ET.Element:
        root = ET.fromstring("""
        <mujoco>
          <worldbody>
            <body name="robot_a" pos="1.0 2.0 0.0">
              <geom name="robot_a_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
            <body name="robot_b" pos="3.0 4.0 0.0">
              <geom name="robot_b_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)
        return root.find("worldbody")

    def test_apply_position_change(self) -> None:
        worldbody = self._make_scene_xml()
        change = PositionChange(
            equipment_id="robot_a",
            new_position=(5.0, 6.0, 0.0),
        )
        _apply_position_change(worldbody, change)
        body = worldbody.find("body[@name='robot_a']")
        assert body.get("pos") == "5.000 6.000 0.000"

    def test_remove_body(self) -> None:
        worldbody = self._make_scene_xml()
        assert worldbody.find("body[@name='robot_b']") is not None
        _remove_body(worldbody, "robot_b")
        assert worldbody.find("body[@name='robot_b']") is None

    def test_remove_nonexistent_is_safe(self) -> None:
        worldbody = self._make_scene_xml()
        _remove_body(worldbody, "nonexistent")
        # Should not raise
