"""Tests for MuJoCo simulation runner."""

import mujoco
import numpy as np

from backend.app.models.equipment import EquipmentEntry, MjcfSource
from backend.app.models.recommendation import WorkflowStep
from backend.app.models.simulation import StepResult
from backend.app.services.controllers import IKEngine
from backend.app.services.simulator import (
    _execute_pick,
    _execute_place,
    _find_nearest_object,
    _resolve_gripper_body,
    _sim_conveyor,
    compute_metrics,
)

# Simple arm with freejoint object and weld constraint for pick/place tests
_ARM_SCENE_XML = """\
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.3"/>
      <body name="link2" pos="0 0 0.3">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.3"/>
        <site name="ee_site" pos="0 0 0.3"/>
      </body>
    </body>
    <body name="work_obj" pos="0.2 0 0.5">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100"/>
    <position name="a2" joint="j2" kp="100"/>
  </actuator>
  <equality>
    <weld name="grasp_weld" body1="link2" body2="work_obj" active="false"/>
  </equality>
</mujoco>
"""

# Conveyor belt scene with a box on a flat belt surface
_CONVEYOR_SCENE_XML = """\
<mujoco>
  <worldbody>
    <body name="conv1" pos="0 0 0">
      <geom name="conv1_belt" type="box" size="1.0 0.2 0.01" pos="0 0 0"/>
    </body>
    <body name="parcel" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""

# Scene with no sites for testing missing equipment error
_NO_SITE_SCENE_XML = """\
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.01"/>
  </worldbody>
</mujoco>
"""


def _load(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model/data pair from an XML string."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


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


class TestExecutePick:
    """Tests for pick action with IK and grasp."""

    def test_pick_attaches_object(self) -> None:
        model, data = _load(_ARM_SCENE_XML)
        ik = IKEngine(model, data, "ee_site")
        target = np.array([0.3, 0.0, 0.9])
        step = WorkflowStep(
            order=1,
            action="pick",
            equipment_id="arm",
            target="pick_pos",
            duration_s=2.0,
        )
        result = _execute_pick(model, data, ik, target, "arm", step)
        assert result.success
        assert result.duration_s == 2.0
        # Weld should be activated
        assert model.eq_active0[0] == 1

    def test_pick_unreachable_returns_error(self) -> None:
        model, data = _load(_ARM_SCENE_XML)
        ik = IKEngine(model, data, "ee_site")
        far_target = np.array([5.0, 5.0, 5.0])
        step = WorkflowStep(
            order=1,
            action="pick",
            equipment_id="arm",
            target="far",
            duration_s=2.0,
        )
        result = _execute_pick(model, data, ik, far_target, "arm", step)
        assert not result.success
        assert "IK failed" in result.error


class TestExecutePlace:
    """Tests for place action with detach."""

    def test_place_detaches_object(self) -> None:
        model, data = _load(_ARM_SCENE_XML)
        ik = IKEngine(model, data, "ee_site")
        pick_target = np.array([0.3, 0.0, 0.9])
        pick_step = WorkflowStep(
            order=1,
            action="pick",
            equipment_id="arm",
            target="pick_pos",
            duration_s=2.0,
        )
        _execute_pick(model, data, ik, pick_target, "arm", pick_step)
        assert model.eq_active0[0] == 1

        place_target = np.array([0.2, 0.0, 0.8])
        place_step = WorkflowStep(
            order=2,
            action="place",
            equipment_id="arm",
            target="place_pos",
            duration_s=2.0,
        )
        result = _execute_place(model, data, ik, place_target, "arm", place_step)
        assert result.success
        assert model.eq_active0[0] == 0


class TestConveyor:
    """Tests for conveyor belt simulation."""

    def test_conveyor_moves_object(self) -> None:
        model, data = _load(_CONVEYOR_SCENE_XML)
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel")
        initial_x = float(data.xpos[obj_id, 0])

        step = WorkflowStep(
            order=1,
            action="transport",
            equipment_id="conv1",
            target="dest",
            duration_s=1.0,
            params={"speed": 0.5},
        )
        result = _sim_conveyor(model, data, step)
        assert result.success

        mujoco.mj_forward(model, data)
        final_x = float(data.xpos[obj_id, 0])
        # Object should have moved in x direction
        assert final_x != initial_x


class TestMissingEquipment:
    """Tests for graceful handling of missing equipment."""

    def test_missing_ee_site_returns_error(self) -> None:
        model, data = _load(_NO_SITE_SCENE_XML)
        step = WorkflowStep(
            order=1,
            action="pick",
            equipment_id="nonexistent_arm",
            target="pos",
            duration_s=2.0,
        )
        entry = EquipmentEntry(
            id="nonexistent_arm",
            name="Missing Arm",
            type="manipulator",
            mjcf_source=MjcfSource(),
        )
        from backend.app.services.simulator import _scripted_manipulation

        result = _scripted_manipulation(model, data, step, (0.0, 0.0, 0.0), entry)
        assert not result.success
        assert "No EE site" in result.error


class TestFindNearestObject:
    """Tests for free-joint object proximity search."""

    def test_finds_nearest_freejoint_body(self) -> None:
        model, data = _load(_ARM_SCENE_XML)
        target = np.array([0.2, 0.0, 0.5])
        name = _find_nearest_object(model, data, target)
        assert name == "work_obj"

    def test_returns_none_when_no_freejoint(self) -> None:
        model, data = _load(_NO_SITE_SCENE_XML)
        target = np.array([0.0, 0.0, 0.0])
        name = _find_nearest_object(model, data, target)
        assert name is None


class TestResolveGripperBody:
    """Tests for gripper body resolution."""

    def test_finds_last_jointed_body(self) -> None:
        model, _ = _load(_ARM_SCENE_XML)
        name = _resolve_gripper_body(model, "arm")
        assert name == "link2"

    def test_returns_none_for_no_joints(self) -> None:
        model, _ = _load(_NO_SITE_SCENE_XML)
        name = _resolve_gripper_body(model, "x")
        assert name is None
