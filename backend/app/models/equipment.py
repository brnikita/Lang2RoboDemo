"""Equipment catalog models."""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "EquipmentEntry",
    "MjcfSource",
    "PlacementRules",
]


class MjcfSource(BaseModel):
    """Source of the MJCF/URDF model for an equipment entry.

    Args:
        menagerie_id: MuJoCo Menagerie model ID (e.g. "franka_emika_panda").
        robot_descriptions_id: robot_descriptions package model ID.
        urdf_url: Direct URL to a URDF file for download.
    """

    menagerie_id: str | None = None
    robot_descriptions_id: str | None = None
    urdf_url: str | None = None


class PlacementRules(BaseModel):
    """Constraints for equipment placement in the scene.

    Args:
        min_zone_m2: Minimum zone area required for this equipment.
        constraints: Additional placement constraints as key-value pairs.
    """

    min_zone_m2: float = Field(default=0.0, ge=0)
    constraints: dict[str, str] = Field(default_factory=dict)


class EquipmentEntry(BaseModel):
    """A single equipment entry in the knowledge-base catalog.

    Args:
        id: Unique equipment identifier.
        name: Human-readable equipment name.
        type: Equipment type determining simulation behavior.
        specs: Technical specifications (reach, payload, dimensions, etc.).
        mjcf_source: Source for downloading the MJCF/URDF model.
        price_usd: Approximate price in USD.
        purchase_url: URL to purchase the equipment.
        placement_rules: Placement constraints.
    """

    id: str
    name: str
    type: Literal["manipulator", "conveyor", "camera", "fixture"]
    specs: dict[str, float | str | int] = Field(default_factory=dict)
    mjcf_source: MjcfSource
    price_usd: float | None = None
    purchase_url: str | None = None
    placement_rules: PlacementRules | None = None
