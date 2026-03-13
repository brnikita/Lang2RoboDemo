"""Tests for equipment model downloader."""

from pathlib import Path

from backend.app.models.equipment import EquipmentEntry, MjcfSource
from backend.app.services.downloader import (
    _create_placeholder,
    _estimate_equipment_size,
    _menagerie_to_description_id,
)


class TestMenagerieMapping:
    """Tests for menagerie ID to robot_descriptions mapping."""

    def test_known_mappings(self) -> None:
        assert _menagerie_to_description_id("franka_emika_panda") == "panda_mj_description"
        assert _menagerie_to_description_id("universal_robots_ur5e") == "ur5e_mj_description"

    def test_unknown_returns_none(self) -> None:
        assert _menagerie_to_description_id("unknown_robot") is None


class TestPlaceholder:
    """Tests for placeholder MJCF generation."""

    def test_creates_mjcf_file(self, tmp_path: Path) -> None:
        entry = EquipmentEntry(
            id="test_conveyor",
            name="Test Conveyor",
            type="conveyor",
            specs={"length_m": 1.0, "width_m": 0.2, "height_m": 0.85},
            mjcf_source=MjcfSource(),
        )
        result = _create_placeholder(entry, tmp_path)
        mjcf = tmp_path / "test_conveyor.xml"
        assert mjcf.exists()
        content = mjcf.read_text()
        assert "test_conveyor" in content

    def test_size_from_specs(self) -> None:
        entry = EquipmentEntry(
            id="conv",
            name="Conv",
            type="conveyor",
            specs={"length_m": 1.0, "width_m": 0.2, "height_m": 0.85},
            mjcf_source=MjcfSource(),
        )
        size = _estimate_equipment_size(entry)
        assert abs(size[0] - 0.5) < 1e-6  # half length
        assert abs(size[1] - 0.1) < 1e-6  # half width

    def test_default_size(self) -> None:
        entry = EquipmentEntry(
            id="cam",
            name="Camera",
            type="camera",
            specs={"fov_deg": 60},
            mjcf_source=MjcfSource(),
        )
        size = _estimate_equipment_size(entry)
        assert size == (0.15, 0.15, 0.15)
