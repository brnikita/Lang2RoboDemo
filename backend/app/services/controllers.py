"""IK engine, grasp manager, and end-effector resolution for MuJoCo scenes."""

import logging

import mujoco
import numpy as np

__all__ = ["IKEngine", "GraspManager", "find_ee_site"]

logger = logging.getLogger(__name__)

_IK_ALPHA = 0.5
_EXCLUDED_ACTUATOR_KEYWORDS = ("conveyor", "belt")

_EE_SITE_NAMES: dict[str, str] = {
    "ufactory_xarm7": "attachment_site",
    "franka_emika_panda": "end_effector",
    "universal_robots_ur5e": "attachment_site",
    "aloha": "left/gripper",
    "kinova_gen3": "pinch_site",
    "sawyer": "attachment_site",
    "kuka_iiwa_14": "attachment_site",
}

_FALLBACK_SITE_NAMES = (
    "attachment_site",
    "end_effector",
    "grip_site",
    "tool_site",
    "ee_site",
    "pinch_site",
)


class IKEngine:
    """Jacobian-based IK controller for any articulated manipulator."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str,
    ) -> None:
        """Initialize IK engine for a specific end-effector site.

        Args:
            model: MuJoCo model.
            data: MuJoCo simulation data.
            site_name: Name of the end-effector site to control.

        Raises:
            ValueError: If site_name is not found in the model.
        """
        self._model = model
        self._data = data
        self._site_id = _resolve_site_id(model, site_name)
        self._act_indices = _find_robot_actuators(model)
        self._jacp = np.zeros((3, model.nv))

    def reach_target(
        self,
        target_pos: np.ndarray,
        max_steps: int = 500,
        tolerance: float = 0.01,
    ) -> bool:
        """Drive end-effector to target position via Jacobian transpose IK.

        Args:
            target_pos: Desired (x, y, z) position.
            max_steps: Maximum IK iterations.
            tolerance: Position error threshold in meters.

        Returns:
            True if target was reached within tolerance.
        """
        target = np.asarray(target_pos, dtype=np.float64)
        for _ in range(max_steps):
            error_norm = self._ik_step(target)
            if error_norm < tolerance:
                return True
        return False

    def _ik_step(self, target_pos: np.ndarray) -> float:
        """Execute a single IK iteration.

        Args:
            target_pos: Desired (x, y, z) position.

        Returns:
            Position error norm after the step.
        """
        mujoco.mj_forward(self._model, self._data)
        error = target_pos - self._data.site_xpos[self._site_id]

        self._jacp[:] = 0
        mujoco.mj_jacSite(
            self._model,
            self._data,
            self._jacp,
            None,
            self._site_id,
        )
        dq = _compute_joint_velocity(self._jacp, error)
        self._apply_control(dq)

        mujoco.mj_step(self._model, self._data)
        return float(np.linalg.norm(error))

    def _apply_control(self, dq: np.ndarray) -> None:
        """Apply joint velocity increments to actuator controls.

        Args:
            dq: Joint velocity vector (nv,).
        """
        for idx in self._act_indices:
            jnt_id = self._model.actuator_trnid[idx, 0]
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            self._data.ctrl[idx] = self._data.qpos[qpos_adr] + dq[self._model.jnt_dofadr[jnt_id]]


def _compute_joint_velocity(
    jacp: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    """Compute damped Jacobian transpose joint velocity.

    Args:
        jacp: Position Jacobian (3 x nv).
        error: Cartesian position error (3,).

    Returns:
        Joint velocity vector (nv,).
    """
    return _IK_ALPHA * jacp.T @ error


def _resolve_site_id(model: mujoco.MjModel, site_name: str) -> int:
    """Resolve a site name to its MuJoCo ID.

    Args:
        model: MuJoCo model.
        site_name: Site name string.

    Returns:
        Integer site ID.

    Raises:
        ValueError: If site not found.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        msg = f"Site '{site_name}' not found in model"
        raise ValueError(msg)
    return site_id


def _find_robot_actuators(model: mujoco.MjModel) -> list[int]:
    """Find actuator indices belonging to the robot (not conveyors).

    Args:
        model: MuJoCo model.

    Returns:
        List of actuator indices.
    """
    indices: list[int] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and _is_robot_actuator(name):
            indices.append(i)
    return indices


def _is_robot_actuator(name: str) -> bool:
    """Check whether an actuator name belongs to the robot.

    Args:
        name: Actuator name.

    Returns:
        True if actuator is not a conveyor/belt.
    """
    lower = name.lower()
    return not any(kw in lower for kw in _EXCLUDED_ACTUATOR_KEYWORDS)


class GraspManager:
    """Weld-constraint based grasp for attaching objects to gripper."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gripper_body_name: str,
    ) -> None:
        """Initialize grasp manager for a gripper body.

        Args:
            model: MuJoCo model.
            data: MuJoCo simulation data.
            gripper_body_name: Name of the gripper body.

        Raises:
            ValueError: If gripper body is not found.
        """
        self._model = model
        self._data = data
        self._gripper_id = _resolve_body_id(model, gripper_body_name)
        self._weld_idx = _find_inactive_weld(model)

    def attach(self, object_body_name: str) -> bool:
        """Attach an object to the gripper via weld constraint.

        Args:
            object_body_name: Name of the object body to grasp.

        Returns:
            True if weld was activated successfully.
        """
        if self._weld_idx is None:
            logger.warning("No weld constraint available for grasping")
            return False
        return self._activate_weld(object_body_name)

    def _activate_weld(self, object_body_name: str) -> bool:
        """Activate the weld constraint between gripper and object.

        Args:
            object_body_name: Name of the object body.

        Returns:
            True if activation succeeded.
        """
        obj_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_body_name,
        )
        if obj_id < 0:
            logger.warning("Object body '%s' not found", object_body_name)
            return False

        idx = self._weld_idx
        self._model.eq_obj1id[idx] = self._gripper_id
        self._model.eq_obj2id[idx] = obj_id
        self._model.eq_active0[idx] = 1
        return True

    def detach(self) -> None:
        """Deactivate the weld constraint, releasing the object."""
        if self._weld_idx is not None:
            self._model.eq_active0[self._weld_idx] = 0


def _resolve_body_id(model: mujoco.MjModel, body_name: str) -> int:
    """Resolve a body name to its MuJoCo ID.

    Args:
        model: MuJoCo model.
        body_name: Body name string.

    Returns:
        Integer body ID.

    Raises:
        ValueError: If body not found.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        msg = f"Body '{body_name}' not found in model"
        raise ValueError(msg)
    return body_id


def _find_inactive_weld(model: mujoco.MjModel) -> int | None:
    """Find the first inactive weld equality constraint.

    Args:
        model: MuJoCo model.

    Returns:
        Constraint index or None if no weld found.
    """
    for i in range(model.neq):
        is_weld = model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD
        if is_weld and not model.eq_active0[i]:
            return i
    return None


def find_ee_site(model: mujoco.MjModel, equipment_id: str) -> str:
    """Find the end-effector site name for a given robot.

    Checks known mappings first, then falls back to common site name
    patterns, and finally returns the last site in the model.

    Args:
        model: MuJoCo model.
        equipment_id: Equipment identifier (e.g. "franka_emika_panda").

    Returns:
        Name of the end-effector site.

    Raises:
        ValueError: If no site could be found in the model.
    """
    known = _lookup_known_site(model, equipment_id)
    if known is not None:
        return known

    fallback = _search_fallback_sites(model)
    if fallback is not None:
        return fallback

    return _last_site_name(model)


def _lookup_known_site(
    model: mujoco.MjModel,
    equipment_id: str,
) -> str | None:
    """Look up a known end-effector site mapping.

    Args:
        model: MuJoCo model.
        equipment_id: Equipment identifier.

    Returns:
        Site name if found and valid, else None.
    """
    site_name = _EE_SITE_NAMES.get(equipment_id)
    if site_name is None:
        return None
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return site_name if site_id >= 0 else None


def _search_fallback_sites(model: mujoco.MjModel) -> str | None:
    """Search for common end-effector site name patterns.

    Args:
        model: MuJoCo model.

    Returns:
        First matching site name, or None.
    """
    for name in _FALLBACK_SITE_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return name
    return None


def _last_site_name(model: mujoco.MjModel) -> str:
    """Return the name of the last site in the model.

    Args:
        model: MuJoCo model.

    Returns:
        Name of the last site.

    Raises:
        ValueError: If the model has no sites.
    """
    if model.nsite == 0:
        msg = "Model has no sites"
        raise ValueError(msg)
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        model.nsite - 1,
    )
    if not name:
        msg = "Last site has no name"
        raise ValueError(msg)
    return name
