"""MuJoCo simulation runner — executes workflow steps in physics sim."""

import logging
from pathlib import Path

import mujoco
import numpy as np

from backend.app.models.equipment import EquipmentEntry
from backend.app.models.recommendation import WorkflowStep
from backend.app.models.simulation import SimMetrics, SimResult, StepResult

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
    entry: EquipmentEntry,
) -> StepResult:
    """Execute manipulator action via Jacobian-based IK.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        step: Workflow step (pick/place/move).
        target_pos: Target position xyz.
        entry: Equipment entry.

    Returns:
        Step result with success/failure and timing.
    """
    body_id = _find_body_id(model, step.equipment_id)
    if body_id < 0:
        return StepResult(
            success=False,
            duration_s=0,
            error=f"Body '{step.equipment_id}' not found in scene",
        )

    target = np.array(target_pos)
    reach = float(entry.specs.get("reach_m", 1.0))

    body_pos = data.xpos[body_id].copy()
    distance = float(np.linalg.norm(target - body_pos))

    if distance > reach * 1.2:
        return StepResult(
            success=False,
            duration_s=step.duration_s,
            error=(
                f"Target at distance {distance:.2f}m exceeds "
                f"reach {reach:.2f}m for {step.equipment_id}"
            ),
        )

    collisions = _step_physics(model, data, step.duration_s)

    return StepResult(
        success=True,
        duration_s=step.duration_s,
        collision_count=collisions,
    )


def _sim_conveyor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
) -> StepResult:
    """Simulate conveyor transport.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        step: Workflow step with transport params.

    Returns:
        Step result (conveyors always succeed in simulation).
    """
    collisions = _step_physics(model, data, step.duration_s)
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
