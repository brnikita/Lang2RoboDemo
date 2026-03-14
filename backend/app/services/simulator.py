"""MuJoCo simulation runner — executes workflow steps in physics sim."""

import logging
from pathlib import Path

import mujoco
import numpy as np

from backend.app.models.equipment import EquipmentEntry
from backend.app.models.recommendation import WorkflowStep
from backend.app.models.simulation import SimMetrics, SimResult, StepResult
from backend.app.services.controllers import GraspManager, IKEngine, find_ee_site

__all__ = ["run_simulation", "compute_metrics"]

logger = logging.getLogger(__name__)


async def run_simulation(
    scene_path: Path,
    workflow: list[WorkflowStep],
    catalog: dict[str, EquipmentEntry],
    target_positions: dict[str, tuple[float, float, float]],
) -> SimResult:
    """Run full simulation of workflow in MuJoCo.

    Args:
        scene_path: Path to MJCF scene file.
        workflow: Ordered workflow steps.
        catalog: Equipment catalog for type dispatch.
        target_positions: Named target positions (name → xyz).

    Returns:
        Simulation result with per-step results and metrics.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    results: list[StepResult] = []
    for step in sorted(workflow, key=lambda s: s.order):
        result = _execute_step(model, data, step, catalog, target_positions)
        results.append(result)

    metrics = compute_metrics(results)
    return SimResult(steps=results, metrics=metrics)


def compute_metrics(results: list[StepResult]) -> SimMetrics:
    """Compute aggregate metrics from step results.

    Args:
        results: Per-step results.

    Returns:
        Aggregate metrics.
    """
    if not results:
        return SimMetrics(cycle_time_s=0, success_rate=0)

    total_time = sum(r.duration_s for r in results)
    successes = sum(1 for r in results if r.success)
    collisions = sum(r.collision_count for r in results)
    failed = [i for i, r in enumerate(results) if not r.success]

    return SimMetrics(
        cycle_time_s=total_time,
        success_rate=successes / len(results),
        collision_count=collisions,
        failed_steps=failed,
    )


def _execute_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    catalog: dict[str, EquipmentEntry],
    target_positions: dict[str, tuple[float, float, float]],
) -> StepResult:
    """Execute one workflow step via the appropriate controller.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        step: Workflow step to execute.
        catalog: Equipment catalog.
        target_positions: Named targets.

    Returns:
        Step execution result.
    """
    if step.action == "wait":
        return _sim_wait(model, data, step.duration_s)

    if step.equipment_id is None:
        return StepResult(
            success=False,
            duration_s=0,
            error="No equipment for non-wait step",
        )

    entry = catalog.get(step.equipment_id)
    if not entry:
        return StepResult(
            success=False,
            duration_s=0,
            error=f"Equipment '{step.equipment_id}' not in catalog",
        )

    target_pos = target_positions.get(step.target)
    if target_pos is None:
        return StepResult(
            success=False,
            duration_s=0,
            error=f"Target '{step.target}' not in target_positions",
        )

    try:
        if entry.type == "manipulator":
            return _scripted_manipulation(
                model,
                data,
                step,
                target_pos,
                entry,
            )
        if entry.type == "conveyor":
            return _sim_conveyor(model, data, step)
        if entry.type == "camera":
            return _sim_camera_inspect(
                model,
                data,
                step,
                target_pos,
                entry,
            )
    except Exception as exc:
        logger.error("Step %d failed: %s", step.order, exc)
        return StepResult(
            success=False,
            duration_s=0,
            error=str(exc),
        )

    return StepResult(
        success=False,
        duration_s=0,
        error=f"Unsupported type '{entry.type}' for action '{step.action}'",
    )


def _sim_wait(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration_s: float,
) -> StepResult:
    """Simulate waiting by stepping physics forward.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        duration_s: Wait duration in seconds.

    Returns:
        Successful step result.
    """
    n_steps = int(duration_s / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    return StepResult(success=True, duration_s=duration_s)


def _scripted_manipulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    target_pos: tuple[float, float, float],
    _entry: EquipmentEntry,
) -> StepResult:
    """Execute manipulator action using IK controller and grasp.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        step: Workflow step (pick/place/move).
        target_pos: Target position xyz.
        _entry: Equipment entry (reserved for future reach checks).

    Returns:
        Step result with success/failure and timing.
    """
    equipment_id = step.equipment_id
    try:
        ee_site = find_ee_site(model, equipment_id)
    except ValueError:
        return StepResult(
            success=False,
            duration_s=step.duration_s,
            error=f"No EE site for {equipment_id}",
        )

    ik = IKEngine(model, data, ee_site)
    target = np.array(target_pos)

    if step.action == "pick":
        return _execute_pick(model, data, ik, target, equipment_id, step)
    if step.action == "place":
        return _execute_place(model, data, ik, target, equipment_id, step)
    return _execute_move(model, data, ik, target, step)


def _execute_pick(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: IKEngine,
    target_pos: np.ndarray,
    equipment_id: str,
    step: WorkflowStep,
) -> StepResult:
    """Execute a pick action: reach, grasp, and lift.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ik: IK engine for the manipulator.
        target_pos: Target position to pick from.
        equipment_id: Equipment performing the pick.
        step: Workflow step.

    Returns:
        Step result.
    """
    if not ik.reach_target(target_pos, max_steps=1000, tolerance=0.05):
        return StepResult(
            success=False,
            duration_s=step.duration_s,
            error=f"IK failed to reach pick target for {equipment_id}",
        )

    obj_name = _find_nearest_object(model, data, target_pos)
    if obj_name is None:
        return StepResult(
            success=False,
            duration_s=step.duration_s,
            error="No free object found near pick target",
        )

    gripper_body = _resolve_gripper_body(model, equipment_id)
    if gripper_body is None:
        return StepResult(
            success=False,
            duration_s=step.duration_s,
            error=f"No gripper body found for {equipment_id}",
        )

    try:
        gm = GraspManager(model, data, gripper_body)
    except ValueError as exc:
        return StepResult(success=False, duration_s=step.duration_s, error=str(exc))

    attached = gm.attach(obj_name)

    lift_pos = target_pos.copy()
    lift_pos[2] += 0.1
    ik.reach_target(lift_pos, max_steps=200, tolerance=0.05)

    collisions = _step_physics(model, data, 0.1)

    return StepResult(
        success=attached,
        duration_s=step.duration_s,
        collision_count=collisions,
    )


def _execute_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: IKEngine,
    target_pos: np.ndarray,
    equipment_id: str,
    step: WorkflowStep,
) -> StepResult:
    """Execute a place action: reach target, release object, settle.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ik: IK engine for the manipulator.
        target_pos: Target position to place at.
        equipment_id: Equipment performing the place.
        step: Workflow step.

    Returns:
        Step result.
    """
    if not ik.reach_target(target_pos, max_steps=1000, tolerance=0.05):
        return StepResult(
            success=False,
            duration_s=step.duration_s,
            error=f"IK failed to reach place target for {equipment_id}",
        )

    _deactivate_all_welds(model)

    collisions = _step_physics(model, data, 0.2)

    return StepResult(
        success=True,
        duration_s=step.duration_s,
        collision_count=collisions,
    )


def _execute_move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: IKEngine,
    target_pos: np.ndarray,
    step: WorkflowStep,
) -> StepResult:
    """Execute a move action: reach target position.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ik: IK engine for the manipulator.
        target_pos: Target position to move to.
        step: Workflow step.

    Returns:
        Step result.
    """
    reached = ik.reach_target(target_pos, max_steps=1000, tolerance=0.05)
    collisions = _step_physics(model, data, 0.1)

    return StepResult(
        success=reached,
        duration_s=step.duration_s,
        collision_count=collisions,
        error=None if reached else "IK failed to reach move target",
    )


def _sim_conveyor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
) -> StepResult:
    """Simulate conveyor transport with belt forces on contacting objects.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        step: Workflow step with transport params.

    Returns:
        Step result with collision count.
    """
    speed = step.params.get("speed", 0.1) if step.params else 0.1
    speed = float(speed)
    belt_geom_name = f"{step.equipment_id}_belt"
    belt_geom_id = _find_geom_id(model, belt_geom_name)

    duration_steps = int(step.duration_s / model.opt.timestep)
    collisions = 0
    for _ in range(min(duration_steps, 5000)):
        if belt_geom_id >= 0:
            _apply_belt_forces(model, data, belt_geom_id, speed)
        mujoco.mj_step(model, data)
        collisions += data.ncon

    return StepResult(
        success=True,
        duration_s=step.duration_s,
        collision_count=collisions,
    )


def _sim_camera_inspect(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    target_pos: tuple[float, float, float],
    entry: EquipmentEntry,
) -> StepResult:
    """Simulate camera inspection — check if target is in FOV.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        step: Workflow step.
        target_pos: Target position to inspect.
        entry: Camera equipment entry.

    Returns:
        Step result with visibility check.
    """
    camera_id = _find_camera_id(model, step.equipment_id)
    if camera_id < 0:
        return StepResult(
            success=False,
            duration_s=0.1,
            error=f"Camera '{step.equipment_id}' not found in scene",
        )

    visible = _check_camera_fov(
        model,
        data,
        camera_id,
        target_pos,
        entry,
    )
    return StepResult(
        success=visible,
        duration_s=0.1,
        error=None if visible else (f"Target '{step.target}' not in camera FOV"),
    )


def _find_body_id(model: mujoco.MjModel, name: str) -> int:
    """Find body ID by name.

    Args:
        model: MuJoCo model.
        name: Body name.

    Returns:
        Body ID or -1 if not found.
    """
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    except Exception:
        return -1


def _find_camera_id(model: mujoco.MjModel, name: str) -> int:
    """Find camera ID by name.

    Args:
        model: MuJoCo model.
        name: Camera name.

    Returns:
        Camera ID or -1 if not found.
    """
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    except Exception:
        return -1


def _find_geom_id(model: mujoco.MjModel, name: str) -> int:
    """Find geom ID by name.

    Args:
        model: MuJoCo model.
        name: Geom name.

    Returns:
        Geom ID or -1 if not found.
    """
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    except Exception:
        return -1


def _find_nearest_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
) -> str | None:
    """Find the nearest free-joint body to a target position.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        target_pos: Reference position for proximity search.

    Returns:
        Name of the closest free-joint body, or None.
    """
    best_name: str | None = None
    best_dist = float("inf")

    for i in range(model.nbody):
        jnt_adr = model.body_jntadr[i]
        if jnt_adr < 0:
            continue
        if model.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_pos = data.xpos[i]
        dist = float(np.linalg.norm(body_pos - target_pos))
        if dist < best_dist:
            best_dist = dist
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                best_name = name
                best_dist = dist

    return best_name


def _resolve_gripper_body(
    model: mujoco.MjModel,
    _equipment_id: str,
) -> str | None:
    """Find the deepest body in the equipment's kinematic chain.

    Uses the last body in the model that has a non-free joint.

    Args:
        model: MuJoCo model.
        _equipment_id: Equipment identifier (reserved for prefix matching).

    Returns:
        Gripper body name, or None.
    """
    candidate: str | None = None
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name:
            continue
        if model.body_jntadr[i] < 0:
            continue
        jnt_type = model.jnt_type[model.body_jntadr[i]]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        candidate = name
    return candidate


def _deactivate_all_welds(model: mujoco.MjModel) -> None:
    """Deactivate all active weld equality constraints (release grasped objects).

    Args:
        model: MuJoCo model.
    """
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD and model.eq_active0[i]:
            model.eq_active0[i] = 0


def _apply_belt_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    belt_geom_id: int,
    speed: float,
) -> None:
    """Apply lateral force to bodies in contact with the belt geom.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        belt_geom_id: MuJoCo geom ID of the belt surface.
        speed: Desired belt speed in m/s.
    """
    for c_idx in range(data.ncon):
        contact = data.contact[c_idx]
        geom1, geom2 = contact.geom1, contact.geom2
        if geom1 != belt_geom_id and geom2 != belt_geom_id:
            continue
        other_geom = geom2 if geom1 == belt_geom_id else geom1
        body_id = model.geom_bodyid[other_geom]
        if body_id > 0:
            data.xfrc_applied[body_id, 0] = speed * 10.0


def _check_camera_fov(
    _model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    target_pos: tuple[float, float, float],
    entry: EquipmentEntry,
) -> bool:
    """Check if a target position is within camera's field of view.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        camera_id: MuJoCo camera ID.
        target_pos: Target position.
        entry: Camera equipment entry.

    Returns:
        True if target is approximately visible.
    """
    cam_pos = data.cam_xpos[camera_id]
    target = np.array(target_pos)
    distance = float(np.linalg.norm(target - cam_pos))

    fov_deg = float(entry.specs.get("fov_deg", 60))
    max_visible_distance = 5.0

    if distance > max_visible_distance:
        return False

    to_target = target - cam_pos
    cam_dir = data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    cos_angle = float(np.dot(to_target, -cam_dir) / (np.linalg.norm(to_target) + 1e-8))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    return angle_deg < fov_deg / 2


def _step_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration_s: float,
) -> int:
    """Step physics forward and count collisions.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        duration_s: Duration to simulate.

    Returns:
        Number of contacts/collisions detected.
    """
    n_steps = int(duration_s / model.opt.timestep)
    total_contacts = 0

    for _ in range(min(n_steps, 1000)):
        mujoco.mj_step(model, data)
        total_contacts += data.ncon

    return total_contacts
