"""Tests for Pydantic data models."""

import pytest

from backend.app.models.equipment import EquipmentEntry, MjcfSource
from backend.app.models.iteration import PositionChange, SceneCorrections
from backend.app.models.recommendation import (
    EquipmentPlacement,
    ExpectedMetrics,
    Recommendation,
    WorkflowStep,
    WorkObject,
)
from backend.app.models.simulation import SimMetrics, SimResult, StepResult
from backend.app.models.space import (
    Dimensions,
    Door,
    ExistingEquipment,
    ReferenceCalibration,
    SceneAnalysis,
    SpaceModel,
    Window,
    Zone,
)


class TestSpaceModels:
    """Tests for space-related Pydantic models."""

    def test_reference_calibration_valid(self) -> None:
        cal = ReferenceCalibration(
            point_a=(0.0, 0.0, 0.0),
            point_b=(1.0, 0.0, 0.0),
            real_distance_m=0.9,
        )
        assert cal.real_distance_m == 0.9

    def test_reference_calibration_rejects_zero_distance(self) -> None:
        with pytest.raises(ValueError):
            ReferenceCalibration(
                point_a=(0, 0, 0), point_b=(1, 0, 0), real_distance_m=0.0
            )

    def test_dimensions_valid(self) -> None:
        dims = Dimensions(width_m=6.0, length_m=5.0, ceiling_m=3.0, area_m2=30.0)
        assert dims.area_m2 == 30.0

    def test_zone_valid(self) -> None:
        zone = Zone(
            name="workstation_1",
            polygon=[(0, 0), (2, 0), (2, 2), (0, 2)],
            area_m2=4.0,
        )
        assert zone.name == "workstation_1"

    def test_door_valid(self) -> None:
        door = Door(position=(1.0, 0.0), width_m=0.9)
        assert door.width_m == 0.9

    def test_window_valid(self) -> None:
        window = Window(position=(3.0, 0.0), width_m=1.2)
        assert window.width_m == 1.2

    def test_existing_equipment_valid(self) -> None:
        eq = ExistingEquipment(
            name="printer_1",
            category="3d_printer",
            position=(1.0, 2.0, 0.85),
            confidence=0.92,
        )
        assert eq.confidence == 0.92

    def test_existing_equipment_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError):
            ExistingEquipment(
                name="x", category="y", position=(0, 0, 0), confidence=1.5
            )

    def test_scene_analysis_defaults_empty(self) -> None:
        analysis = SceneAnalysis()
        assert analysis.zones == []
        assert analysis.existing_equipment == []

    def test_space_model_complete(self, sample_space_model: SpaceModel) -> None:
        assert sample_space_model.dimensions.width_m == 6.0
        assert sample_space_model.reconstruction.mjcf_path.name == "scene.xml"


class TestEquipmentModels:
    """Tests for equipment catalog models."""

    def test_equipment_entry_valid(self) -> None:
        entry = EquipmentEntry(
            id="franka_panda",
            name="Franka Emika Panda",
            type="manipulator",
            specs={"reach_m": 0.855, "payload_kg": 3.0},
            mjcf_source=MjcfSource(menagerie_id="franka_emika_panda"),
            price_usd=30000.0,
        )
        assert entry.type == "manipulator"

    def test_equipment_entry_camera(self) -> None:
        entry = EquipmentEntry(
            id="cam_overhead",
            name="Overhead Camera",
            type="camera",
            mjcf_source=MjcfSource(),
        )
        assert entry.type == "camera"


class TestRecommendationModels:
    """Tests for recommendation models."""

    def test_workflow_step_valid(self) -> None:
        step = WorkflowStep(
            order=1, action="pick", equipment_id="franka", target="table_1", duration_s=3.0
        )
        assert step.action == "pick"

    def test_workflow_step_wait_no_equipment(self) -> None:
        step = WorkflowStep(
            order=5, action="wait", target="next_item", duration_s=10.0
        )
        assert step.equipment_id is None

    def test_work_object_valid(self) -> None:
        obj = WorkObject(
            name="box",
            shape="box",
            size=(0.1, 0.1, 0.05),
            mass_kg=0.2,
            position=(1.0, 1.0, 0.85),
            count=5,
        )
        assert obj.count == 5

    def test_recommendation_complete(self) -> None:
        rec = Recommendation(
            equipment=[
                EquipmentPlacement(
                    equipment_id="franka_panda",
                    position=(2.0, 1.5, 0.0),
                    purpose="Pick and place",
                    zone="main",
                )
            ],
            target_positions={"table_1": (1.0, 1.0, 0.85)},
            workflow_steps=[
                WorkflowStep(
                    order=1, action="pick", equipment_id="franka_panda",
                    target="table_1", duration_s=3.0,
                )
            ],
            expected_metrics=ExpectedMetrics(
                cycle_time_s=15.0, throughput_per_hour=240.0
            ),
        )
        assert len(rec.equipment) == 1


class TestSimulationModels:
    """Tests for simulation result models."""

    def test_step_result_success(self) -> None:
        result = StepResult(success=True, duration_s=2.5)
        assert result.collision_count == 0

    def test_step_result_failure(self) -> None:
        result = StepResult(
            success=False, duration_s=0.0, error="IK solver failed"
        )
        assert result.error is not None

    def test_sim_metrics_valid(self) -> None:
        metrics = SimMetrics(
            cycle_time_s=15.0, success_rate=0.8, collision_count=2, failed_steps=[3]
        )
        assert metrics.success_rate == 0.8

    def test_sim_result_complete(self) -> None:
        result = SimResult(
            steps=[StepResult(success=True, duration_s=3.0)],
            metrics=SimMetrics(cycle_time_s=3.0, success_rate=1.0),
        )
        assert len(result.steps) == 1


class TestIterationModels:
    """Tests for iteration models."""

    def test_position_change(self) -> None:
        change = PositionChange(
            equipment_id="franka",
            new_position=(2.5, 1.5, 0.0),
            new_orientation_deg=90.0,
        )
        assert change.new_orientation_deg == 90.0

    def test_scene_corrections_empty(self) -> None:
        corrections = SceneCorrections()
        assert corrections.position_changes is None
        assert corrections.add_equipment is None

    def test_scene_corrections_with_changes(self) -> None:
        corrections = SceneCorrections(
            position_changes=[
                PositionChange(equipment_id="franka", new_position=(3.0, 2.0, 0.0))
            ],
            remove_equipment=["old_cam"],
        )
        assert len(corrections.position_changes) == 1
        assert "old_cam" in corrections.remove_equipment
