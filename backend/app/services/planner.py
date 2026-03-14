"""Claude-powered recommendation and iteration planning."""

import json
import logging

from backend.app.core.claude import ClaudeClient
from backend.app.core.config import get_settings
from backend.app.core.prompts import load_prompt
from backend.app.models.equipment import EquipmentEntry
from backend.app.models.recommendation import Recommendation
from backend.app.models.space import SpaceModel
from backend.app.services.catalog import load_equipment_catalog

__all__ = [
    "generate_recommendation",
    "format_recommendation_context",
    "parse_and_validate",
]

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


async def generate_recommendation(
    client: ClaudeClient,
    space: SpaceModel,
    scenario: str,
) -> Recommendation:
    """Generate an automation plan via Claude.

    Args:
        client: Claude API client.
        space: Room model.
        scenario: User's text description of desired automation.

    Returns:
        Validated recommendation with equipment from catalog.

    Raises:
        ValueError: If Claude returns invalid data after retries.
    """
    catalog = load_equipment_catalog()
    system_prompt = load_prompt("recommendation")
    context = format_recommendation_context(space, scenario, catalog)

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.send_message(
                system=system_prompt,
                messages=[{"role": "user", "content": context}],
                model=get_settings().planning_model,
            )
            return parse_and_validate(response, catalog)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "Recommendation attempt %d/%d failed: %s",
                attempt + 1,
                _MAX_RETRIES + 1,
                exc,
            )
            context = _build_retry_context(context, exc)

    raise ValueError(
        f"Failed to generate valid recommendation after {_MAX_RETRIES + 1} attempts: {last_error}"
    )


def format_recommendation_context(
    space: SpaceModel,
    scenario: str,
    catalog: dict[str, EquipmentEntry],
) -> str:
    """Format input context for Claude recommendation request.

    Args:
        space: Room model.
        scenario: User scenario text.
        catalog: Equipment catalog.

    Returns:
        Formatted context string.
    """
    catalog_summary = _format_catalog(catalog)
    space_json = space.model_dump_json(indent=2, exclude={"reconstruction"})

    return (
        f"## User Scenario\n\n{scenario}\n\n"
        f"## Room Model (SpaceModel)\n\n```json\n{space_json}\n```\n\n"
        f"## Available Equipment Catalog\n\n{catalog_summary}\n"
    )


def parse_and_validate(
    response: str,
    catalog: dict[str, EquipmentEntry],
) -> Recommendation:
    """Parse Claude response and validate against catalog.

    Args:
        response: Raw response text from Claude.
        catalog: Equipment catalog for ID validation.

    Returns:
        Validated Recommendation.

    Raises:
        ValueError: If response is unparseable or contains invalid IDs.
        KeyError: If equipment ID not found in catalog.
    """
    json_str = _extract_json(response)
    rec = Recommendation.model_validate_json(json_str)
    _validate_equipment_ids(rec, catalog)
    _validate_workflow_refs(rec)
    _override_prices(rec, catalog)
    return rec


def _validate_equipment_ids(
    rec: Recommendation,
    catalog: dict[str, EquipmentEntry],
) -> None:
    """Validate all equipment IDs reference the catalog.

    Args:
        rec: Recommendation to validate.
        catalog: Equipment catalog.

    Raises:
        KeyError: If any equipment_id is not in catalog.
    """
    for placement in rec.equipment:
        if placement.equipment_id not in catalog:
            raise KeyError(
                f"Equipment '{placement.equipment_id}' not in catalog. "
                f"Available: {', '.join(sorted(catalog.keys()))}"
            )


def _validate_workflow_refs(rec: Recommendation) -> None:
    """Validate workflow step references are consistent.

    Args:
        rec: Recommendation to validate.

    Raises:
        ValueError: If step references non-existent equipment or target.
    """
    placed_ids = {p.equipment_id for p in rec.equipment}
    for step in rec.workflow_steps:
        if step.equipment_id and step.equipment_id not in placed_ids:
            raise ValueError(
                f"Workflow step {step.order} references '{step.equipment_id}' not in equipment list"
            )
        if step.target not in rec.target_positions:
            raise ValueError(
                f"Workflow step {step.order} references "
                f"target '{step.target}' not in target_positions"
            )


def _override_prices(
    rec: Recommendation,
    catalog: dict[str, EquipmentEntry],
) -> None:
    """Ensure prices come from catalog, not from Claude.

    Args:
        rec: Recommendation (modified in place).
        catalog: Authoritative price source.
    """
    # Prices are stored in catalog, not in Recommendation model
    # This is a no-op but documents the design decision


def _format_catalog(
    catalog: dict[str, EquipmentEntry],
) -> str:
    """Format equipment catalog for Claude context.

    Args:
        catalog: Equipment entries.

    Returns:
        Formatted catalog text.
    """
    lines: list[str] = []
    for entry_id, entry in sorted(catalog.items()):
        specs_str = ", ".join(f"{k}: {v}" for k, v in entry.specs.items())
        lines.append(f"- **{entry_id}** ({entry.type}): {entry.name} — {specs_str}")
    return "\n".join(lines)


def _build_retry_context(original: str, error: Exception) -> str:
    """Build context for retry with error feedback.

    Args:
        original: Original context.
        error: The error that occurred.

    Returns:
        Updated context with error info.
    """
    return (
        f"{original}\n\n"
        f"## IMPORTANT: Previous attempt failed\n\n"
        f"Error: {error}\n\n"
        f"Fix the issue and return valid JSON."
    )


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
