"""Tests for recommendation planner."""

import json

import pytest

from backend.app.models.equipment import EquipmentEntry, MjcfSource
from backend.app.models.recommendation import (
    EquipmentPlacement,
    ExpectedMetrics,
    Recommendation,
    WorkflowStep,
    WorkObject,
)
from backend.app.services.planner import (
    _extract_json,
    format_recommendation_context,
    parse_and_validate,
)


def _make_catalog() -> dict[str, EquipmentEntry]:
    """Create a minimal test catalog."""
    return {
        "franka_emika_panda": EquipmentEntry(
            id="franka_emika_panda",
            name="Franka Emika Panda",
            type="manipulator",
            specs={"reach_m": 0.855},
            mjcf_source=MjcfSource(menagerie_id="franka_emika_panda"),
        ),
        "conveyor_500mm": EquipmentEntry(
            id="conveyor_500mm",
            name="Belt Conveyor 500mm",
            type="conveyor",
            specs={"length_m": 0.5},
            mjcf_source=MjcfSource(),
        ),
        "camera_overhead": EquipmentEntry(
            id="camera_overhead",
            name="Overhead Camera",
            type="camera",
            specs={"fov_deg": 60},
            mjcf_source=MjcfSource(),
        ),
    }


def _make_valid_recommendation_json() -> str:
    """Create valid recommendation JSON for testing."""
    rec = Recommendation(
        equipment=[
            EquipmentPlacement(
                equipment_id="franka_emika_panda",
                position=(2.0, 1.5, 0.0),
                purpose="Pick and place",
                zone="main",
            ),
        ],
        work_objects=[
            WorkObject(
                name="item",
                shape="box",
                size=(0.05, 0.05, 0.04),
                mass_kg=0.1,
                position=(1.0, 1.0, 0.85),
                count=3,
            ),
        ],
        target_positions={
            "table_1": (1.0, 1.0, 0.85),
            "conveyor_start": (3.0, 2.0, 0.85),
        },
        workflow_steps=[
            WorkflowStep(
                order=1,
                action="pick",
                equipment_id="franka_emika_panda",
                target="table_1",
                duration_s=3.0,
            ),
            WorkflowStep(
                order=2,
                action="place",
                equipment_id="franka_emika_panda",
                target="conveyor_start",
                duration_s=3.0,
            ),
        ],
        expected_metrics=ExpectedMetrics(
            cycle_time_s=6.0,
            throughput_per_hour=600,
        ),
        text_plan="Pick items from table, place on conveyor.",
    )
    return rec.model_dump_json()


class TestParseAndValidate:
    """Tests for response parsing and validation."""

    def test_valid_json_passes(self) -> None:
        catalog = _make_catalog()
        json_str = _make_valid_recommendation_json()
        rec = parse_and_validate(json_str, catalog)
        assert len(rec.equipment) == 1
        assert rec.equipment[0].equipment_id == "franka_emika_panda"

    def test_json_in_code_block(self) -> None:
        catalog = _make_catalog()
        json_str = _make_valid_recommendation_json()
        wrapped = f"```json\n{json_str}\n```"
        rec = parse_and_validate(wrapped, catalog)
        assert len(rec.equipment) == 1

    def test_invalid_equipment_id_raises(self) -> None:
        catalog = _make_catalog()
        json_str = _make_valid_recommendation_json()
        data = json.loads(json_str)
        data["equipment"][0]["equipment_id"] = "nonexistent_robot"
        with pytest.raises(KeyError, match="nonexistent_robot"):
            parse_and_validate(json.dumps(data), catalog)

    def test_invalid_workflow_target_raises(self) -> None:
        catalog = _make_catalog()
        json_str = _make_valid_recommendation_json()
        data = json.loads(json_str)
        data["workflow_steps"][0]["target"] = "nonexistent_target"
        with pytest.raises(ValueError, match="nonexistent_target"):
            parse_and_validate(json.dumps(data), catalog)

    def test_workflow_equipment_not_placed_raises(self) -> None:
        catalog = _make_catalog()
        json_str = _make_valid_recommendation_json()
        data = json.loads(json_str)
        data["workflow_steps"][0]["equipment_id"] = "conveyor_500mm"
        with pytest.raises(ValueError, match="conveyor_500mm"):
            parse_and_validate(json.dumps(data), catalog)


class TestFormatContext:
    """Tests for context formatting."""

    def test_includes_scenario(self, sample_space_model) -> None:
        catalog = _make_catalog()
        ctx = format_recommendation_context(
            sample_space_model,
            "Test scenario text",
            catalog,
        )
        assert "Test scenario text" in ctx

    def test_includes_catalog_ids(self, sample_space_model) -> None:
        catalog = _make_catalog()
        ctx = format_recommendation_context(
            sample_space_model,
            "Test",
            catalog,
        )
        assert "franka_emika_panda" in ctx
        assert "conveyor_500mm" in ctx

    def test_includes_room_dimensions(self, sample_space_model) -> None:
        catalog = _make_catalog()
        ctx = format_recommendation_context(
            sample_space_model,
            "Test",
            catalog,
        )
        assert "6.0" in ctx  # width_m


class TestExtractJson:
    """Tests for JSON extraction."""

    def test_plain_json(self) -> None:
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_code_block(self) -> None:
        assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
