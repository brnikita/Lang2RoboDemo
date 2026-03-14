"""Tests for iteration module."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.app.models.equipment import EquipmentEntry, MjcfSource
from backend.app.models.iteration import EquipmentReplacement, PositionChange
from backend.app.models.recommendation import EquipmentPlacement
from backend.app.models.simulation import SimMetrics
from backend.app.services.iteration import (
    _apply_position_change,
    _dispatch_add_equipment,
    _euler_to_deg,
    _is_converged,
    _next_scene_path,
    _remove_body,
    _replace_body,
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


class TestReplaceBody:
    """Tests for equipment replacement with real model dispatch."""

    def _make_full_scene(self) -> ET.Element:
        return ET.fromstring("""
        <mujoco>
          <worldbody>
            <body name="old_robot" pos="1.0 2.0 0.0" euler="0 0 0.7854">
              <geom name="old_robot_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    def _make_entry(self, eq_type: str = "conveyor") -> EquipmentEntry:
        return EquipmentEntry(
            id="new_conv_1",
            name="Test Conveyor",
            type=eq_type,
            specs={"length_m": 1.0, "width_m": 0.3, "height_m": 0.8},
            mjcf_source=MjcfSource(),
        )

    def test_replace_body_conveyor(self) -> None:
        root = self._make_full_scene()
        worldbody = root.find("worldbody")
        replacement = EquipmentReplacement(
            old_equipment_id="old_robot",
            new_equipment_id="new_conv_1",
            reason="testing",
        )
        entry = self._make_entry("conveyor")
        _replace_body(root, worldbody, replacement, entry, None, Path("/tmp"))

        assert worldbody.find("body[@name='old_robot']") is None
        new_body = worldbody.find("body[@name='new_conv_1']")
        assert new_body is not None
        assert new_body.get("pos") == "1.000 2.000 0.000"
        belt = new_body.find("geom[@name='new_conv_1_belt']")
        assert belt is not None

    def test_replace_body_fixture(self) -> None:
        root = self._make_full_scene()
        worldbody = root.find("worldbody")
        replacement = EquipmentReplacement(
            old_equipment_id="old_robot",
            new_equipment_id="new_fixture_1",
            reason="testing",
        )
        entry = EquipmentEntry(
            id="new_fixture_1",
            name="Test Fixture",
            type="fixture",
            specs={"length_m": 0.5, "width_m": 0.5, "height_m": 1.0},
            mjcf_source=MjcfSource(),
        )
        _replace_body(root, worldbody, replacement, entry, None, Path("/tmp"))

        new_body = worldbody.find("body[@name='new_fixture_1']")
        assert new_body is not None
        geom = new_body.find("geom[@name='new_fixture_1_geom']")
        assert geom is not None
        assert geom.get("type") == "box"

    def test_replace_nonexistent_is_safe(self) -> None:
        root = self._make_full_scene()
        worldbody = root.find("worldbody")
        replacement = EquipmentReplacement(
            old_equipment_id="does_not_exist",
            new_equipment_id="new_conv_1",
            reason="testing",
        )
        entry = self._make_entry()
        _replace_body(root, worldbody, replacement, entry, None, Path("/tmp"))
        # old body still there, no new body added
        assert worldbody.find("body[@name='old_robot']") is not None
        assert worldbody.find("body[@name='new_conv_1']") is None

    def test_replace_preserves_position(self) -> None:
        root = self._make_full_scene()
        worldbody = root.find("worldbody")
        replacement = EquipmentReplacement(
            old_equipment_id="old_robot",
            new_equipment_id="new_conv_1",
            reason="testing",
        )
        entry = self._make_entry()
        _replace_body(root, worldbody, replacement, entry, None, Path("/tmp"))
        new_body = worldbody.find("body[@name='new_conv_1']")
        assert new_body.get("pos") == "1.000 2.000 0.000"
        euler_z = float(new_body.get("euler", "0 0 0").split()[2])
        assert abs(euler_z - 0.7854) < 0.01


class TestDispatchAddEquipment:
    """Tests for type-dispatched equipment addition."""

    def _make_root(self) -> ET.Element:
        return ET.fromstring("<mujoco><worldbody/></mujoco>")

    def test_add_camera(self) -> None:
        root = self._make_root()
        worldbody = root.find("worldbody")
        placement = EquipmentPlacement(
            equipment_id="cam_1",
            position=(1.0, 2.0, 0.0),
            orientation_deg=0.0,
            purpose="test",
            zone="default",
        )
        entry = EquipmentEntry(
            id="cam_1",
            name="Test Camera",
            type="camera",
            specs={"mounting_height_m": 2.0},
            mjcf_source=MjcfSource(),
        )
        _dispatch_add_equipment(
            root,
            worldbody,
            placement,
            "cam_1",
            entry,
            {},
            Path("/tmp"),
        )
        body = worldbody.find("body[@name='cam_1']")
        assert body is not None
        housing = body.find("geom[@name='cam_1_housing']")
        assert housing is not None

    def test_add_conveyor(self) -> None:
        root = self._make_root()
        worldbody = root.find("worldbody")
        placement = EquipmentPlacement(
            equipment_id="conv_1",
            position=(2.0, 3.0, 0.0),
            orientation_deg=90.0,
            purpose="test",
            zone="default",
        )
        entry = EquipmentEntry(
            id="conv_1",
            name="Belt Conveyor",
            type="conveyor",
            specs={"length_m": 1.0, "width_m": 0.3, "height_m": 0.8},
            mjcf_source=MjcfSource(),
        )
        _dispatch_add_equipment(
            root,
            worldbody,
            placement,
            "conv_1",
            entry,
            {},
            Path("/tmp"),
        )
        body = worldbody.find("body[@name='conv_1']")
        assert body is not None
        assert body.find("geom[@name='conv_1_belt']") is not None
        assert body.find("geom[@name='conv_1_frame']") is not None

    def test_output_is_valid_xml(self) -> None:
        root = self._make_root()
        worldbody = root.find("worldbody")
        placement = EquipmentPlacement(
            equipment_id="fix_1",
            position=(0.5, 0.5, 0.0),
            orientation_deg=0.0,
            purpose="test",
            zone="default",
        )
        entry = EquipmentEntry(
            id="fix_1",
            name="Jig",
            type="fixture",
            specs={"length_m": 0.4, "width_m": 0.4, "height_m": 0.6},
            mjcf_source=MjcfSource(),
        )
        _dispatch_add_equipment(
            root,
            worldbody,
            placement,
            "fix_1",
            entry,
            {},
            Path("/tmp"),
        )
        # Should produce valid XML that can be serialized
        xml_str = ET.tostring(root, encoding="unicode")
        reparsed = ET.fromstring(xml_str)
        assert reparsed.find("worldbody/body[@name='fix_1']") is not None


class TestEulerToDeg:
    """Tests for euler string to degrees conversion."""

    def test_zero(self) -> None:
        assert _euler_to_deg("0 0 0") == 0.0

    def test_90_degrees(self) -> None:
        rad = math.radians(90.0)
        result = _euler_to_deg(f"0 0 {rad:.4f}")
        assert abs(result - 90.0) < 0.1

    def test_short_string(self) -> None:
        assert _euler_to_deg("0 0") == 0.0
