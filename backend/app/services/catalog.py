"""Equipment catalog loader and validator."""

import json
import logging
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.models.equipment import EquipmentEntry

__all__ = ["load_equipment_catalog", "validate_equipment_id"]

logger = logging.getLogger(__name__)

_catalog_cache: dict[str, EquipmentEntry] | None = None


def load_equipment_catalog() -> dict[str, EquipmentEntry]:
    """Load and cache all equipment entries from the knowledge-base JSONs.

    Returns:
        Mapping of equipment_id to EquipmentEntry.

    Raises:
        FileNotFoundError: If knowledge-base directory doesn't exist.
    """
    global _catalog_cache  # noqa: PLW0603
    if _catalog_cache is not None:
        return _catalog_cache

    settings = get_settings()
    equipment_dir = settings.KNOWLEDGE_BASE_DIR / "equipment"
    _catalog_cache = _load_from_directory(equipment_dir)
    logger.info("Loaded %d equipment entries", len(_catalog_cache))
    return _catalog_cache


def _load_from_directory(directory: Path) -> dict[str, EquipmentEntry]:
    """Parse all JSON files in a directory into EquipmentEntry objects.

    Args:
        directory: Path to the equipment JSON files.

    Returns:
        Mapping of equipment_id to validated EquipmentEntry.
    """
    catalog: dict[str, EquipmentEntry] = {}
    for json_file in sorted(directory.glob("*.json")):
        entries = json.loads(json_file.read_text(encoding="utf-8"))
        for raw in entries:
            entry = EquipmentEntry.model_validate(raw)
            catalog[entry.id] = entry
    return catalog


def validate_equipment_id(
    equipment_id: str,
    catalog: dict[str, EquipmentEntry] | None = None,
) -> EquipmentEntry:
    """Validate that an equipment ID exists in the catalog.

    Args:
        equipment_id: ID to validate.
        catalog: Optional pre-loaded catalog (loads default if None).

    Returns:
        The matching EquipmentEntry.

    Raises:
        KeyError: If equipment_id is not found in the catalog.
    """
    if catalog is None:
        catalog = load_equipment_catalog()
    if equipment_id not in catalog:
        available = ", ".join(sorted(catalog.keys()))
        raise KeyError(f"Equipment '{equipment_id}' not found. Available: {available}")
    return catalog[equipment_id]
