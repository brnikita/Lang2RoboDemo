"""Claude Vision scene analysis — photos → structured SceneAnalysis."""

import json
import logging
from pathlib import Path

from backend.app.core.claude import ClaudeClient
from backend.app.core.prompts import load_prompt
from backend.app.models.space import (
    SceneAnalysis,
    SceneReconstruction,
    SpaceModel,
)

__all__ = ["analyze_scene", "build_space_model"]

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


async def analyze_scene(
    client: ClaudeClient,
    photos: list[Path],
    reconstruction: SceneReconstruction,
) -> SceneAnalysis:
    """Analyze room photos + reconstruction data via Claude Vision.

    Args:
        client: Claude API client.
        photos: List of room photo file paths.
        reconstruction: DISCOVERSE reconstruction result.

    Returns:
        Structured scene analysis with zones, equipment, doors, windows.

    Raises:
        ValueError: If Claude returns unparseable response after retries.
    """
    system_prompt = load_prompt("vision_analysis")
    dims = reconstruction.dimensions
    text = _format_analysis_request(dims)

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.send_vision_message(
                system=system_prompt,
                images=photos,
                text=text,
            )
            return _parse_analysis_response(response)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "Analysis attempt %d/%d failed: %s",
                attempt + 1, _MAX_RETRIES + 1, exc,
            )

    raise ValueError(
        f"Failed to parse scene analysis after {_MAX_RETRIES + 1} attempts: "
        f"{last_error}"
    )


def build_space_model(
    reconstruction: SceneReconstruction,
    analysis: SceneAnalysis,
) -> SpaceModel:
    """Merge reconstruction and analysis into a complete SpaceModel.

    Args:
        reconstruction: DISCOVERSE reconstruction data.
        analysis: Claude Vision analysis results.

    Returns:
        Complete room model for simulation.
    """
    return SpaceModel(
        dimensions=reconstruction.dimensions,
        zones=analysis.zones,
        existing_equipment=analysis.existing_equipment,
        doors=analysis.doors,
        windows=analysis.windows,
        reconstruction=reconstruction,
    )


def _format_analysis_request(dims: "Dimensions") -> str:
    """Format the text portion of the vision analysis request.

    Args:
        dims: Room dimensions from reconstruction.

    Returns:
        Formatted request text.
    """
    from backend.app.models.space import Dimensions

    return (
        f"Room dimensions from 3D reconstruction:\n"
        f"  Width: {dims.width_m:.2f}m\n"
        f"  Length: {dims.length_m:.2f}m\n"
        f"  Ceiling: {dims.ceiling_m:.2f}m\n"
        f"  Area: {dims.area_m2:.2f}m²\n\n"
        f"Analyze these photos and identify:\n"
        f"1. Functional zones (name, polygon, area)\n"
        f"2. Existing equipment (name, category, position, confidence)\n"
        f"3. Doors (position, width)\n"
        f"4. Windows (position, width)\n\n"
        f"Return ONLY valid JSON matching the SceneAnalysis schema."
    )


def _parse_analysis_response(response: str) -> SceneAnalysis:
    """Parse Claude's response into a SceneAnalysis model.

    Args:
        response: Raw response text from Claude.

    Returns:
        Parsed SceneAnalysis.

    Raises:
        ValueError: If response cannot be parsed.
    """
    json_str = _extract_json(response)
    return SceneAnalysis.model_validate_json(json_str)


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks.

    Args:
        text: Raw text that may contain JSON in code blocks.

    Returns:
        Extracted JSON string.
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
