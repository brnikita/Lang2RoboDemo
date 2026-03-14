"""Iterative improvement loop — Claude corrections + re-simulation."""

import json
import logging
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.app.core.claude import ClaudeClient
from backend.app.core.config import get_settings
from backend.app.core.prompts import load_prompt
from backend.app.models.equipment import EquipmentEntry
from backend.app.models.iteration import (
    EquipmentReplacement,
    IterationLog,
    PositionChange,
    SceneCorrections,
)
from backend.app.models.recommendation import EquipmentPlacement, Recommendation
from backend.app.models.simulation import SimMetrics, SimResult
from backend.app.services.catalog import validate_equipment_id
from backend.app.services.downloader import download_equipment_model
from backend.app.services.scene import (
    _add_camera_body_to_scene,
    _add_conveyor_to_scene,
    _add_fixture_to_scene,
    _add_manipulator_to_scene,
    _format_pos,
)
from backend.app.services.simulator import run_simulation

__all__ = [
    "iterate_once",
    "run_iteration_loop",
    "apply_corrections",
]

logger = logging.getLogger(__name__)

_SUCCESS_THRESHOLD = 0.95
_MAX_RETRIES = 2


async def run_iteration_loop(
    scene_path: Path,
    recommendation: Recommendation,
    catalog: dict[str, EquipmentEntry],
    client: ClaudeClient,
    max_iterations: int = 5,
) -> tuple[SimResult, list[IterationLog]]:
    """Run the full iteration improvement loop.

    Args:
        scene_path: Path to initial MJCF scene.
        recommendation: Current recommendation.
        catalog: Equipment catalog.
        client: Claude API client.
        max_iterations: Maximum iteration count.

    Returns:
        Final simulation result and iteration history.
    """
    history: list[IterationLog] = []
    current_scene = scene_path

    result = await run_simulation(
        current_scene,
        recommendation.workflow_steps,
        catalog,
        recommendation.target_positions,
    )

    if _is_converged(result.metrics):
        return result, history

    for i in range(max_iterations):
        logger.info("Iteration %d: success_rate=%.2f", i + 1, result.metrics.success_rate)

        corrections = await iterate_once(
            current_scene,
            result.metrics,
            history,
            catalog,
            client,
        )

        new_scene = _next_scene_path(current_scene)
        await apply_corrections(
            current_scene,
            corrections,
            catalog,
            new_scene,
        )

        history.append(
            IterationLog(
                iteration=i + 1,
                metrics=result.metrics,
                corrections_applied=corrections,
            )
        )

        current_scene = new_scene
        result = await run_simulation(
            current_scene,
            recommendation.workflow_steps,
            catalog,
            recommendation.target_positions,
        )

        if _is_converged(result.metrics):
            logger.info("Converged at iteration %d", i + 1)
            break

    return result, history


async def iterate_once(
    scene_path: Path,
    metrics: SimMetrics,
    history: list[IterationLog],
    catalog: dict[str, EquipmentEntry],
    client: ClaudeClient,
) -> SceneCorrections:
    """Run one iteration: send metrics to Claude, get corrections.

    Args:
        scene_path: Current scene MJCF path.
        metrics: Latest simulation metrics.
        history: Previous iteration logs.
        catalog: Equipment catalog.
        client: Claude API client.

    Returns:
        Proposed corrections.

    Raises:
        ValueError: If Claude returns unparseable response.
    """
    system_prompt = load_prompt("iteration")
    context = _format_iteration_context(
        scene_path,
        metrics,
        history,
        catalog,
    )

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.send_message(
                system=system_prompt,
                messages=[{"role": "user", "content": context}],
                model=get_settings().planning_model,
            )
            return _parse_corrections(response, catalog)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "Iteration parse attempt %d failed: %s",
                attempt + 1,
                exc,
            )

    raise ValueError(f"Failed to parse corrections: {last_error}")


async def apply_corrections(
    scene_path: Path,
    corrections: SceneCorrections,
    catalog: dict[str, EquipmentEntry],
    output_path: Path,
) -> Path:
    """Apply corrections to scene MJCF and save as new version.

    Args:
        scene_path: Current scene path.
        corrections: Corrections to apply.
        catalog: Equipment catalog.
        output_path: Path for corrected scene.

    Returns:
        Path to corrected scene file.
    """
    shutil.copy2(scene_path, output_path)
    tree = ET.parse(str(output_path))
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if corrections.position_changes:
        for change in corrections.position_changes:
            _apply_position_change(worldbody, change)

    if corrections.remove_equipment:
        for eq_id in corrections.remove_equipment:
            _remove_body(worldbody, eq_id)

    scene_dir = output_path.parent

    if corrections.replace_equipment:
        for replacement in corrections.replace_equipment:
            entry = validate_equipment_id(replacement.new_equipment_id, catalog)
            model_dir = await download_equipment_model(entry)
            _replace_body(
                root,
                worldbody,
                replacement,
                entry,
                model_dir,
                scene_dir,
            )

    if corrections.add_equipment:
        for placement in corrections.add_equipment:
            entry = catalog.get(placement.equipment_id)
            if entry:
                await _add_equipment_body(
                    root,
                    worldbody,
                    placement,
                    entry,
                    scene_dir,
                )

    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="unicode", xml_declaration=False)
    return output_path


def _is_converged(metrics: SimMetrics) -> bool:
    """Check if simulation meets success criteria.

    Args:
        metrics: Current metrics.

    Returns:
        True if converged.
    """
    return metrics.success_rate >= _SUCCESS_THRESHOLD and metrics.collision_count == 0


def _next_scene_path(current: Path) -> Path:
    """Generate next version scene path.

    Args:
        current: Current scene file path (e.g., v1.xml).

    Returns:
        Next version path (e.g., v2.xml).
    """
    stem = current.stem
    version = int(stem[1:]) + 1 if stem.startswith("v") and stem[1:].isdigit() else 2
    return current.parent / f"v{version}.xml"


def _format_iteration_context(
    scene_path: Path,
    metrics: SimMetrics,
    history: list[IterationLog],
    catalog: dict[str, EquipmentEntry],
) -> str:
    """Format context for Claude iteration request.

    Args:
        scene_path: Current MJCF path.
        metrics: Latest metrics.
        history: Previous iterations.
        catalog: Equipment catalog.

    Returns:
        Formatted context string.
    """
    scene_xml = scene_path.read_text(encoding="utf-8")
    metrics_json = metrics.model_dump_json(indent=2)

    history_text = ""
    if history:
        history_dicts = [h.model_dump() for h in history]
        history_text = json.dumps(history_dicts, indent=2)

    catalog_ids = ", ".join(sorted(catalog.keys()))

    return (
        f"## Current Scene (MJCF)\n\n```xml\n{scene_xml}\n```\n\n"
        f"## Current Metrics\n\n```json\n{metrics_json}\n```\n\n"
        f"## Iteration History\n\n{history_text or 'First iteration'}\n\n"
        f"## Available Equipment IDs\n\n{catalog_ids}\n"
    )


def _parse_corrections(
    response: str,
    catalog: dict[str, EquipmentEntry],
) -> SceneCorrections:
    """Parse and validate corrections from Claude response.

    Args:
        response: Raw response text.
        catalog: Equipment catalog for validation.

    Returns:
        Validated SceneCorrections.
    """
    json_str = _extract_json(response)
    corrections = SceneCorrections.model_validate_json(json_str)

    if corrections.replace_equipment:
        for r in corrections.replace_equipment:
            validate_equipment_id(r.new_equipment_id, catalog)

    if corrections.add_equipment:
        for p in corrections.add_equipment:
            validate_equipment_id(p.equipment_id, catalog)

    return corrections


def _apply_position_change(worldbody: ET.Element, change: PositionChange) -> None:
    """Move a body to a new position in the MJCF tree.

    Args:
        worldbody: Worldbody XML element.
        change: PositionChange with new position.
    """
    for body in worldbody.findall("body"):
        if body.get("name") == change.equipment_id:
            body.set("pos", _format_pos(change.new_position))
            if change.new_orientation_deg is not None:
                euler = f"0 0 {math.radians(change.new_orientation_deg):.4f}"
                body.set("euler", euler)
            return


def _remove_body(worldbody: ET.Element, equipment_id: str) -> None:
    """Remove a body from the scene.

    Args:
        worldbody: Worldbody XML element.
        equipment_id: Body name to remove.
    """
    for body in worldbody.findall("body"):
        if body.get("name") == equipment_id:
            worldbody.remove(body)
            return


def _replace_body(
    root: ET.Element,
    worldbody: ET.Element,
    replacement: EquipmentReplacement,
    new_entry: EquipmentEntry,
    model_dir: Path | None,
    scene_dir: Path,
) -> None:
    """Replace one equipment body with another using real models.

    Args:
        root: Root mujoco element (for includes/actuators).
        worldbody: Worldbody XML element.
        replacement: EquipmentReplacement with old/new IDs.
        new_entry: New equipment catalog entry.
        model_dir: Downloaded model directory (or None).
        scene_dir: Scene directory for relative paths.
    """
    for body in worldbody.findall("body"):
        if body.get("name") == replacement.old_equipment_id:
            pos_str = body.get("pos", "0 0 0")
            euler_str = body.get("euler", "0 0 0")
            worldbody.remove(body)
            break
    else:
        return

    pos_parts = [float(p) for p in pos_str.split()]
    position = (pos_parts[0], pos_parts[1], pos_parts[2])
    orientation = _euler_to_deg(euler_str)

    placement = EquipmentPlacement(
        equipment_id=replacement.new_equipment_id,
        position=position,
        orientation_deg=orientation,
        purpose="replacement",
        zone="default",
    )
    model_dirs = {replacement.new_equipment_id: model_dir} if model_dir else {}

    _dispatch_add_equipment(
        root,
        worldbody,
        placement,
        replacement.new_equipment_id,
        new_entry,
        model_dirs,
        scene_dir,
    )


async def _add_equipment_body(
    root: ET.Element,
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    entry: EquipmentEntry,
    scene_dir: Path,
) -> None:
    """Add a new equipment body using real model by type.

    Args:
        root: Root mujoco element (for includes/actuators).
        worldbody: Worldbody XML element.
        placement: EquipmentPlacement.
        entry: Equipment catalog entry.
        scene_dir: Scene directory for relative paths.
    """
    model_dirs: dict[str, Path] = {}
    if entry.type == "manipulator":
        model_dir = await download_equipment_model(entry)
        model_dirs[placement.equipment_id] = model_dir

    _dispatch_add_equipment(
        root,
        worldbody,
        placement,
        placement.equipment_id,
        entry,
        model_dirs,
        scene_dir,
    )


def _dispatch_add_equipment(
    root: ET.Element,
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    body_name: str,
    entry: EquipmentEntry,
    model_dirs: dict[str, Path],
    scene_dir: Path,
) -> None:
    """Dispatch equipment addition by type, using real models.

    Args:
        root: Root mujoco element.
        worldbody: Worldbody XML element.
        placement: Equipment placement data.
        body_name: Unique body name.
        entry: Equipment catalog entry.
        model_dirs: Model directory mapping.
        scene_dir: Scene directory for relative paths.
    """
    if entry.type == "manipulator":
        _add_manipulator_to_scene(
            root,
            worldbody,
            placement,
            body_name,
            model_dirs,
            scene_dir,
        )
    elif entry.type == "conveyor":
        _add_conveyor_to_scene(root, worldbody, placement, body_name, entry)
    elif entry.type == "camera":
        _add_camera_body_to_scene(worldbody, placement, body_name, entry)
    else:
        _add_fixture_to_scene(worldbody, placement, body_name, entry)


def _euler_to_deg(euler_str: str) -> float:
    """Extract Z-axis rotation in degrees from euler attribute.

    Args:
        euler_str: Space-separated euler angles string (radians).

    Returns:
        Z-axis rotation in degrees.
    """
    parts = euler_str.split()
    if len(parts) >= 3:
        return math.degrees(float(parts[2]))
    return 0.0


def _extract_json(text: str) -> str:
    """Extract JSON from text with code block handling.

    Args:
        text: Raw response text.

    Returns:
        JSON string.
    """
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return text.strip()
