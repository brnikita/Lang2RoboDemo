"""Tests for MJCF scene generation."""

from pathlib import Path

from backend.app.models.equipment import EquipmentEntry, MjcfSource
from backend.app.models.recommendation import (
    EquipmentPlacement,
    ExpectedMetrics,
    Recommendation,
    WorkflowStep,
    WorkObject,
)
from backend.app.services.scene import generate_mjcf_scene


def _make_catalog() -> dict[str, EquipmentEntry]:
    return {
        "franka_emika_panda": EquipmentEntry(
            id="franka_emika_panda",
            name="Franka Emika Panda",
            type="manipulator",
            specs={"reach_m": 0.855},
            mjcf_source=MjcfSource(menagerie_id="franka_emika_panda"),
        ),
        "camera_overhead": EquipmentEntry(
            id="camera_overhead",
            name="Overhead Camera",
            type="camera",
            specs={"fov_deg": 60, "mounting_height_m": 1.5},
            mjcf_source=MjcfSource(),
        ),
    }


def _make_recommendation() -> Recommendation:
    return Recommendation(
        equipment=[
            EquipmentPlacement(
                equipment_id="franka_emika_panda",
                position=(2.0, 1.5, 0.0),
                purpose="Pick and place",
                zone="main",
            ),
            EquipmentPlacement(
                equipment_id="camera_overhead",
                position=(3.0, 2.0, 0.0),
                purpose="Inspection",
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
                count=2,
            ),
        ],
        target_positions={"table_1": (1.0, 1.0, 0.85)},
        workflow_steps=[
            WorkflowStep(
                order=1,
                action="pick",
                equipment_id="franka_emika_panda",
                target="table_1",
                duration_s=3.0,
            ),
        ],
        expected_metrics=ExpectedMetrics(
            cycle_time_s=3.0,
            throughput_per_hour=1200,
        ),
    )


class TestGenerateMjcfScene:
    """Tests for MJCF scene generation."""

    def test_generates_valid_xml(
        self,
        sample_space_model,
        tmp_path: Path,
    ) -> None:
        catalog = _make_catalog()
        recommendation = _make_recommendation()
        output = tmp_path / "scene.xml"

        generate_mjcf_scene(
            sample_space_model,
            recommendation,
            model_dirs={},
            catalog=catalog,
            output_path=output,
        )

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "<mujoco" in content

    def test_contains_work_objects(
        self,
        sample_space_model,
        tmp_path: Path,
    ) -> None:
        catalog = _make_catalog()
        recommendation = _make_recommendation()
        output = tmp_path / "scene.xml"

        generate_mjcf_scene(
            sample_space_model,
            recommendation,
            model_dirs={},
            catalog=catalog,
            output_path=output,
        )

        content = output.read_text(encoding="utf-8")
        assert "item_0" in content
        assert "item_1" in content

    def test_contains_equipment_body(
        self,
        sample_space_model,
        tmp_path: Path,
    ) -> None:
        catalog = _make_catalog()
        recommendation = _make_recommendation()
        output = tmp_path / "scene.xml"

        generate_mjcf_scene(
            sample_space_model,
            recommendation,
            model_dirs={},
            catalog=catalog,
            output_path=output,
        )

        content = output.read_text(encoding="utf-8")
        assert "franka_emika_panda" in content

    def test_contains_camera(
        self,
        sample_space_model,
        tmp_path: Path,
    ) -> None:
        catalog = _make_catalog()
        recommendation = _make_recommendation()
        output = tmp_path / "scene.xml"

        generate_mjcf_scene(
            sample_space_model,
            recommendation,
            model_dirs={},
            catalog=catalog,
            output_path=output,
        )

        content = output.read_text(encoding="utf-8")
        assert "camera_overhead" in content

    def test_work_objects_have_freejoints(
        self,
        sample_space_model,
        tmp_path: Path,
    ) -> None:
        catalog = _make_catalog()
        recommendation = _make_recommendation()
        output = tmp_path / "scene.xml"

        generate_mjcf_scene(
            sample_space_model,
            recommendation,
            model_dirs={},
            catalog=catalog,
            output_path=output,
        )

        content = output.read_text(encoding="utf-8")
        assert "freejoint" in content
