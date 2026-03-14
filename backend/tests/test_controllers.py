"""Tests for IK engine, grasp manager, and end-effector resolution."""

import mujoco
import numpy as np
import pytest

from backend.app.services.controllers import (
    GraspManager,
    IKEngine,
    find_ee_site,
)

SIMPLE_ARM_XML = """\
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
    <body name="target_box" pos="0.2 0 0.5">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100"/>
    <position name="a2" joint="j2" kp="100"/>
  </actuator>
  <equality>
    <weld name="grasp_weld" body1="link2" body2="target_box" active="false"/>
  </equality>
</mujoco>
"""

NO_WELD_XML = """\
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
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100"/>
    <position name="a2" joint="j2" kp="100"/>
  </actuator>
</mujoco>
"""

CUSTOM_SITE_XML = """\
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.3"/>
      <site name="my_custom_site" pos="0 0 0.3"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a" joint="j" kp="100"/>
  </actuator>
</mujoco>
"""


def _load(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model/data pair from an XML string."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class TestIKEngine:
    """Tests for Jacobian-based IK controller."""

    def test_reaches_reachable_target(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        ik = IKEngine(model, data, "ee_site")
        target = np.array([0.3, 0.0, 0.9])
        assert ik.reach_target(target, max_steps=1000, tolerance=0.05)

    def test_fails_unreachable_target(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        ik = IKEngine(model, data, "ee_site")
        target = np.array([5.0, 5.0, 5.0])
        assert not ik.reach_target(target, max_steps=100, tolerance=0.01)

    def test_invalid_site_raises(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        with pytest.raises(ValueError, match="not found"):
            IKEngine(model, data, "nonexistent_site")


class TestGraspManager:
    """Tests for weld-constraint based grasp."""

    def test_attach_activates_weld(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        gm = GraspManager(model, data, "link2")
        assert gm.attach("target_box")
        weld_idx = 0
        assert model.eq_active0[weld_idx] == 1

    def test_detach_deactivates_weld(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        gm = GraspManager(model, data, "link2")
        gm.attach("target_box")
        gm.detach()
        weld_idx = 0
        assert model.eq_active0[weld_idx] == 0

    def test_attach_fails_no_weld(self) -> None:
        model, data = _load(NO_WELD_XML)
        gm = GraspManager(model, data, "link2")
        assert not gm.attach("nonexistent_body")

    def test_attach_fails_bad_object(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        gm = GraspManager(model, data, "link2")
        assert not gm.attach("nonexistent_body")

    def test_invalid_gripper_raises(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        with pytest.raises(ValueError, match="not found"):
            GraspManager(model, data, "nonexistent_body")


class TestFindEeSite:
    """Tests for end-effector site resolution."""

    def test_known_robot_mapping(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        # ee_site matches the fallback list
        result = find_ee_site(model, "unknown_robot")
        assert result == "ee_site"

    def test_fallback_finds_ee_site(self) -> None:
        model, data = _load(SIMPLE_ARM_XML)
        result = find_ee_site(model, "totally_unknown")
        assert result == "ee_site"

    def test_last_resort_custom_site(self) -> None:
        model, data = _load(CUSTOM_SITE_XML)
        # "my_custom_site" is not in known or fallback lists
        result = find_ee_site(model, "some_robot")
        assert result == "my_custom_site"

    def test_no_sites_raises(self) -> None:
        xml = """\
<mujoco>
  <worldbody>
    <geom type="sphere" size="0.1"/>
  </worldbody>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        with pytest.raises(ValueError, match="no sites"):
            find_ee_site(model, "robot")
