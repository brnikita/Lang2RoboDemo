"""Tests for equipment catalog loading and validation."""

import pytest

from backend.app.services.catalog import (
    _load_from_directory,
    validate_equipment_id,
)
from backend.app.core.config import get_settings


class TestCatalogLoader:
    """Tests for catalog loading from JSON files."""

    def test_load_real_catalog(self) -> None:
        settings = get_settings()
        catalog = _load_from_directory(
            settings.KNOWLEDGE_BASE_DIR / "equipment"
        )
        assert len(catalog) > 0
        assert "franka_emika_panda" in catalog

    def test_all_entries_have_required_fields(self) -> None:
        settings = get_settings()
        catalog = _load_from_directory(
            settings.KNOWLEDGE_BASE_DIR / "equipment"
        )
        for entry_id, entry in catalog.items():
            assert entry.id == entry_id
            assert entry.name
            assert entry.type in ("manipulator", "conveyor", "camera", "fixture")

    def test_manipulators_have_reach(self) -> None:
        settings = get_settings()
        catalog = _load_from_directory(
            settings.KNOWLEDGE_BASE_DIR / "equipment"
        )
        manipulators = {
            k: v for k, v in catalog.items() if v.type == "manipulator"
        }
        assert len(manipulators) >= 3
        for entry in manipulators.values():
            assert "reach_m" in entry.specs

    def test_conveyors_have_length(self) -> None:
        settings = get_settings()
        catalog = _load_from_directory(
            settings.KNOWLEDGE_BASE_DIR / "equipment"
        )
        conveyors = {
            k: v for k, v in catalog.items() if v.type == "conveyor"
        }
        assert len(conveyors) >= 1
        for entry in conveyors.values():
            assert "length_m" in entry.specs


class TestValidateEquipmentId:
    """Tests for equipment ID validation."""

    def test_valid_id(self) -> None:
        settings = get_settings()
        catalog = _load_from_directory(
            settings.KNOWLEDGE_BASE_DIR / "equipment"
        )
        entry = validate_equipment_id("franka_emika_panda", catalog)
        assert entry.name == "Franka Emika Panda"

    def test_invalid_id_raises(self) -> None:
        settings = get_settings()
        catalog = _load_from_directory(
            settings.KNOWLEDGE_BASE_DIR / "equipment"
        )
        with pytest.raises(KeyError, match="not_a_real_robot"):
            validate_equipment_id("not_a_real_robot", catalog)
