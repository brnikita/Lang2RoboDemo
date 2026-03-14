"""Equipment model downloader (Menagerie, robot_descriptions, URDF)."""

import logging
import shutil
from pathlib import Path

import httpx

from backend.app.core.config import get_settings
from backend.app.models.equipment import EquipmentEntry
from backend.app.services.catalog import load_equipment_catalog

__all__ = ["download_equipment_models", "download_equipment_model", "find_mjcf_in_dir"]

logger = logging.getLogger(__name__)


async def download_equipment_models(
    equipment_ids: list[str],
) -> dict[str, Path]:
    """Download MJCF/URDF models for all specified equipment.

    Args:
        equipment_ids: List of equipment IDs from the catalog.

    Returns:
        Mapping of equipment_id to local model directory path.
    """
    catalog = load_equipment_catalog()
    models: dict[str, Path] = {}
    for eq_id in equipment_ids:
        entry = catalog[eq_id]
        models[eq_id] = await download_equipment_model(entry)
    return models


async def download_equipment_model(entry: EquipmentEntry) -> Path:
    """Download a single equipment model, using cache if available.

    Args:
        entry: Equipment catalog entry.

    Returns:
        Path to local model directory containing MJCF/URDF files.
    """
    cache_dir = _get_cache_dir(entry.id)
    if _is_cached(cache_dir):
        logger.info("Using cached model for %s", entry.id)
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    source = entry.mjcf_source

    if source.robot_descriptions_id:
        return _fetch_from_robot_descriptions(
            source.robot_descriptions_id,
            cache_dir,
        )
    if source.menagerie_id:
        return _fetch_from_robot_descriptions_menagerie(
            source.menagerie_id,
            cache_dir,
        )
    if source.urdf_url:
        return await _fetch_from_url(source.urdf_url, cache_dir)

    logger.warning("No model source for %s, creating placeholder", entry.id)
    return _create_placeholder(entry, cache_dir)


def _get_cache_dir(equipment_id: str) -> Path:
    """Get the cache directory for an equipment model.

    Args:
        equipment_id: Equipment identifier.

    Returns:
        Cache directory path.
    """
    settings = get_settings()
    return settings.MODELS_DIR / equipment_id


def _is_cached(cache_dir: Path) -> bool:
    """Check if a model is already cached.

    Args:
        cache_dir: Cache directory to check.

    Returns:
        True if cached and contains files.
    """
    if not cache_dir.exists():
        return False
    return any(cache_dir.rglob("*.xml"))


def find_mjcf_in_dir(model_dir: Path) -> Path | None:
    """Find the main MJCF entry point in a model directory.

    Prefers robot-specific XML (e.g. panda.xml, xarm7.xml) over scene.xml,
    because scene.xml is typically a Menagerie wrapper with includes.

    Args:
        model_dir: Directory containing model files.

    Returns:
        Path to main MJCF file, or None if not found.
    """
    if model_dir is None or not model_dir.exists():
        return None
    direct = model_dir / f"{model_dir.name}.xml"
    if direct.exists():
        return direct
    robot_xml = _find_robot_xml(model_dir)
    if robot_xml:
        return robot_xml
    scene = model_dir / "scene.xml"
    if scene.exists():
        return scene
    xmls = sorted(model_dir.glob("*.xml"))
    return xmls[0] if xmls else None


def _find_robot_xml(model_dir: Path) -> Path | None:
    """Find robot-specific XML, excluding wrappers and variants.

    Args:
        model_dir: Model directory.

    Returns:
        Robot XML path or None.
    """
    skip = {"scene.xml", "hand.xml"}
    for xml in sorted(model_dir.glob("*.xml")):
        if xml.name in skip or xml.name.startswith("mjx_"):
            continue
        return xml
    return None


def _fetch_from_robot_descriptions(
    description_id: str,
    cache_dir: Path,
) -> Path:
    """Fetch model from robot_descriptions package.

    Args:
        description_id: robot_descriptions model ID (e.g. "xarm7_mj_description").
        cache_dir: Local cache directory.

    Returns:
        Cache directory path with model files.
    """
    import importlib

    try:
        module = importlib.import_module(f"robot_descriptions.{description_id}")
        if hasattr(module, "MJCF_PATH"):
            src_path = Path(module.MJCF_PATH)
            _copy_model_tree(src_path, cache_dir)
            return cache_dir
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "Failed to import robot_descriptions.%s: %s",
            description_id,
            exc,
        )

    return cache_dir


def _fetch_from_robot_descriptions_menagerie(
    menagerie_id: str,
    cache_dir: Path,
) -> Path:
    """Fetch Menagerie model via robot_descriptions.

    Args:
        menagerie_id: MuJoCo Menagerie directory name.
        cache_dir: Local cache directory.

    Returns:
        Cache directory path with model files.
    """
    desc_id = _menagerie_to_description_id(menagerie_id)
    if desc_id:
        return _fetch_from_robot_descriptions(desc_id, cache_dir)

    logger.warning(
        "No robot_descriptions mapping for menagerie_id %s",
        menagerie_id,
    )
    return cache_dir


def _menagerie_to_description_id(menagerie_id: str) -> str | None:
    """Map menagerie directory name to robot_descriptions ID.

    Args:
        menagerie_id: MuJoCo Menagerie directory name.

    Returns:
        robot_descriptions ID or None.
    """
    mapping = {
        "franka_emika_panda": "panda_mj_description",
        "universal_robots_ur5e": "ur5e_mj_description",
        "ufactory_xarm7": "xarm7_mj_description",
        "aloha": "aloha_mj_description",
        "trs_so_arm100": "so_arm100_mj_description",
        "kinova_gen3": "gen3_mj_description",
        "rethink_robotics_sawyer": "sawyer_mj_description",
        "trossen_vx300s": "widow_mj_description",
    }
    return mapping.get(menagerie_id)


async def _fetch_from_url(url: str, cache_dir: Path) -> Path:
    """Download URDF file from URL.

    Args:
        url: URL to URDF file.
        cache_dir: Local cache directory.

    Returns:
        Cache directory path with downloaded file.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        filename = url.split("/")[-1]
        dest = cache_dir / filename
        dest.write_bytes(response.content)
    return cache_dir


def _copy_model_tree(src_path: Path, cache_dir: Path) -> None:
    """Copy model file and its directory contents to cache.

    Args:
        src_path: Source MJCF/URDF file path.
        cache_dir: Destination cache directory.
    """
    src_dir = src_path.parent
    for item in src_dir.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src_dir)
            dest = cache_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def _create_placeholder(entry: EquipmentEntry, cache_dir: Path) -> Path:
    """Create a placeholder MJCF for equipment without a model source.

    Args:
        entry: Equipment entry.
        cache_dir: Cache directory.

    Returns:
        Cache directory with placeholder MJCF.
    """
    size = _estimate_equipment_size(entry)
    mjcf = f"""<mujoco model="{entry.id}">
  <worldbody>
    <body name="{entry.id}" pos="0 0 0">
      <geom name="{entry.id}_geom" type="box"
            size="{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}"
            rgba="0.5 0.5 0.5 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    (cache_dir / f"{entry.id}.xml").write_text(mjcf, encoding="utf-8")
    return cache_dir


def _estimate_equipment_size(
    entry: EquipmentEntry,
) -> tuple[float, float, float]:
    """Estimate equipment bounding box from specs.

    Args:
        entry: Equipment entry with specs.

    Returns:
        Half-sizes (x, y, z) for MuJoCo box geom.
    """
    specs = entry.specs
    if "length_m" in specs and "width_m" in specs:
        length = float(specs["length_m"])
        width = float(specs["width_m"])
        height = float(specs.get("height_m", 0.85))
        return (length / 2, width / 2, height / 2)
    return (0.15, 0.15, 0.15)
