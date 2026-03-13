"""MJCF scene generation — room + equipment + work objects → final scene."""

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.app.models.equipment import EquipmentEntry
from backend.app.models.recommendation import (
    EquipmentPlacement,
    Recommendation,
    WorkObject,
)
from backend.app.models.space import SpaceModel

__all__ = ["generate_mjcf_scene", "validate_mjcf"]

logger = logging.getLogger(__name__)


def generate_mjcf_scene(
    space: SpaceModel,
    recommendation: Recommendation,
    model_dirs: dict[str, Path],
    catalog: dict[str, EquipmentEntry],
    output_path: Path,
) -> Path:
    """Build complete MJCF scene: room + equipment + work objects.

    Args:
        space: Room model with reconstruction MJCF.
        recommendation: Automation plan with placements.
        model_dirs: Mapping equipment_id → local model directory.
        catalog: Equipment catalog for type information.
        output_path: Path for output MJCF file.

    Returns:
        Path to the generated MJCF file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = _create_base_scene(space)
    worldbody = root.find("worldbody")

    _add_existing_equipment(worldbody, space)
    _add_new_equipment(
        root, worldbody, recommendation.equipment,
        model_dirs, catalog, output_path.parent,
    )
    _add_work_objects(worldbody, recommendation.work_objects)
    _add_cameras(root, recommendation.equipment, catalog)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="unicode", xml_declaration=False)

    # MuJoCo XML parser doesn't allow comments before root element
    return output_path


def validate_mjcf(scene_path: Path) -> bool:
    """Validate that a MJCF file can be loaded by MuJoCo.

    Args:
        scene_path: Path to MJCF file.

    Returns:
        True if scene loads successfully.
    """
    import mujoco

    try:
        mujoco.MjModel.from_xml_path(str(scene_path))
        return True
    except Exception as exc:
        logger.error("MJCF validation failed: %s", exc)
        return False


def _create_base_scene(space: SpaceModel) -> ET.Element:
    """Create base MJCF structure with room geometry.

    Args:
        space: Room model.

    Returns:
        Root mujoco XML element.
    """
    dims = space.dimensions
    root = ET.Element("mujoco", model="lang2robo_scene")

    option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")
    option.set("timestep", "0.002")

    asset = ET.SubElement(root, "asset")
    _add_texture_and_material(asset)
    _include_room_mesh(asset, space)

    worldbody = ET.SubElement(root, "worldbody")
    _add_lighting(worldbody, dims)
    _add_floor(worldbody, dims)
    _add_room_body(worldbody, space)

    return root


def _add_texture_and_material(asset: ET.Element) -> None:
    """Add default textures and materials to asset element.

    Args:
        asset: Asset XML element.
    """
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512",
        "rgb1": "0.8 0.8 0.8", "rgb2": "0.6 0.6 0.6",
    })
    ET.SubElement(asset, "material", {
        "name": "grid_mat", "texture": "grid",
        "texrepeat": "4 4", "reflectance": "0.1",
    })


def _include_room_mesh(
    asset: ET.Element, space: SpaceModel,
) -> None:
    """Include room mesh from DISCOVERSE reconstruction.

    Args:
        asset: Asset XML element.
        space: Room model with reconstruction paths.
    """
    mesh_path = space.reconstruction.mesh_path
    if mesh_path.exists() and _is_valid_mesh(mesh_path):
        ET.SubElement(asset, "mesh", {
            "name": "room_mesh",
            "file": str(mesh_path).replace("\\", "/"),
        })


def _add_lighting(
    worldbody: ET.Element, dims: "Dimensions",
) -> None:
    """Add scene lighting.

    Args:
        worldbody: Worldbody XML element.
        dims: Room dimensions for positioning.
    """
    cx = dims.width_m / 2
    cy = dims.length_m / 2
    ET.SubElement(worldbody, "light", {
        "pos": f"{cx:.2f} {cy:.2f} {dims.ceiling_m:.2f}",
        "dir": "0 0 -1", "diffuse": "1 1 1",
    })


def _add_floor(
    worldbody: ET.Element, dims: "Dimensions",
) -> None:
    """Add floor plane.

    Args:
        worldbody: Worldbody XML element.
        dims: Room dimensions for sizing.
    """
    sx = dims.width_m / 2
    sy = dims.length_m / 2
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane",
        "size": f"{sx:.2f} {sy:.2f} 0.01",
        "material": "grid_mat",
    })


def _add_room_body(
    worldbody: ET.Element, space: SpaceModel,
) -> None:
    """Add room mesh as visual body.

    Args:
        worldbody: Worldbody XML element.
        space: Room model.
    """
    mesh_path = space.reconstruction.mesh_path
    if mesh_path.exists() and _is_valid_mesh(mesh_path):
        body = ET.SubElement(worldbody, "body", {
            "name": "room", "pos": "0 0 0",
        })
        ET.SubElement(body, "geom", {
            "name": "room_visual", "type": "mesh",
            "mesh": "room_mesh", "contype": "0",
            "conaffinity": "0", "rgba": "0.9 0.9 0.9 0.3",
        })


def _add_existing_equipment(
    worldbody: ET.Element, space: SpaceModel,
) -> None:
    """Add existing equipment as static bodies.

    Args:
        worldbody: Worldbody XML element.
        space: Room model with existing equipment.
    """
    for eq in space.existing_equipment:
        pos = f"{eq.position[0]:.3f} {eq.position[1]:.3f} {eq.position[2]:.3f}"
        body = ET.SubElement(worldbody, "body", {
            "name": eq.name, "pos": pos,
        })
        ET.SubElement(body, "geom", {
            "name": f"{eq.name}_geom", "type": "box",
            "size": "0.2 0.2 0.4", "rgba": "0.6 0.4 0.2 1",
        })


def _add_new_equipment(
    root: ET.Element,
    worldbody: ET.Element,
    placements: list[EquipmentPlacement],
    model_dirs: dict[str, Path],
    catalog: dict[str, EquipmentEntry],
    scene_dir: Path,
) -> None:
    """Add new equipment from recommendation.

    Args:
        root: Root mujoco element (for includes).
        worldbody: Worldbody XML element.
        placements: Equipment placements.
        model_dirs: Model directory mapping.
        catalog: Equipment catalog.
        scene_dir: Scene directory for relative paths.
    """
    for placement in placements:
        entry = catalog.get(placement.equipment_id)
        if not entry:
            continue

        model_dir = model_dirs.get(placement.equipment_id)
        pos = _format_pos(placement.position)
        euler = f"0 0 {math.radians(placement.orientation_deg):.4f}"

        body = ET.SubElement(worldbody, "body", {
            "name": placement.equipment_id,
            "pos": pos, "euler": euler,
        })

        if model_dir and _has_mjcf(model_dir):
            mjcf_file = _find_mjcf(model_dir)
            _inline_include(root, body, mjcf_file, scene_dir)
        else:
            size = _equipment_half_size(entry)
            ET.SubElement(body, "geom", {
                "name": f"{placement.equipment_id}_geom",
                "type": "box",
                "size": f"{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}",
                "rgba": _equipment_color(entry.type),
            })


def _add_work_objects(
    worldbody: ET.Element,
    work_objects: list[WorkObject],
) -> None:
    """Add dynamic work objects for manipulation.

    Args:
        worldbody: Worldbody XML element.
        work_objects: Objects from recommendation.
    """
    for obj in work_objects:
        for i in range(obj.count):
            name = f"{obj.name}_{i}"
            pos = _format_pos(obj.position)

            body = ET.SubElement(worldbody, "body", {
                "name": name, "pos": pos,
            })
            ET.SubElement(body, "freejoint", {"name": f"{name}_joint"})

            geom_attrs = {
                "name": f"{name}_geom",
                "type": obj.shape,
                "mass": f"{obj.mass_kg:.4f}",
                "rgba": "0.2 0.6 0.2 1",
            }
            if obj.shape == "box":
                geom_attrs["size"] = (
                    f"{obj.size[0]/2:.4f} {obj.size[1]/2:.4f} "
                    f"{obj.size[2]/2:.4f}"
                )
            elif obj.shape == "cylinder":
                geom_attrs["size"] = f"{obj.size[0]:.4f} {obj.size[1]/2:.4f}"
            elif obj.shape == "sphere":
                geom_attrs["size"] = f"{obj.size[0]:.4f}"

            ET.SubElement(body, "geom", geom_attrs)


def _add_cameras(
    root: ET.Element,
    placements: list[EquipmentPlacement],
    catalog: dict[str, EquipmentEntry],
) -> None:
    """Add MuJoCo camera elements for camera-type equipment.

    Args:
        root: Root mujoco element.
        placements: Equipment placements.
        catalog: Equipment catalog.
    """
    worldbody = root.find("worldbody")
    for placement in placements:
        entry = catalog.get(placement.equipment_id)
        if not entry or entry.type != "camera":
            continue

        fov = entry.specs.get("fov_deg", 60)
        height = entry.specs.get("mounting_height_m", 1.5)
        pos = (
            f"{placement.position[0]:.3f} "
            f"{placement.position[1]:.3f} "
            f"{float(height):.3f}"
        )
        ET.SubElement(worldbody, "camera", {
            "name": placement.equipment_id,
            "pos": pos, "zaxis": "0 0 -1",
            "fovy": str(int(fov)),
        })


def _format_pos(position: tuple[float, float, float]) -> str:
    """Format 3D position for MJCF attribute.

    Args:
        position: (x, y, z) tuple.

    Returns:
        Space-separated string.
    """
    return f"{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}"


def _has_mjcf(model_dir: Path) -> bool:
    """Check if directory contains MJCF files.

    Args:
        model_dir: Directory to check.

    Returns:
        True if .xml files exist.
    """
    return any(model_dir.glob("*.xml"))


def _find_mjcf(model_dir: Path) -> Path:
    """Find the main MJCF file in a model directory.

    Args:
        model_dir: Model directory.

    Returns:
        Path to main MJCF file.
    """
    xmls = sorted(model_dir.glob("*.xml"))
    for xml in xmls:
        if xml.stem not in ("scene",):
            return xml
    return xmls[0]


def _inline_include(
    root: ET.Element,
    body: ET.Element,
    mjcf_file: Path,
    scene_dir: Path,
) -> None:
    """Include external MJCF model into scene.

    Args:
        root: Root mujoco element.
        body: Parent body element.
        mjcf_file: Path to external MJCF.
        scene_dir: Scene directory for relative paths.
    """
    body.set("childclass", mjcf_file.stem)
    ET.SubElement(body, "include", {
        "file": str(mjcf_file),
    })


def _equipment_half_size(
    entry: EquipmentEntry,
) -> tuple[float, float, float]:
    """Get equipment half-sizes for box geom.

    Args:
        entry: Equipment entry.

    Returns:
        (half_x, half_y, half_z) in meters.
    """
    specs = entry.specs
    if "length_m" in specs and "width_m" in specs:
        return (
            float(specs["length_m"]) / 2,
            float(specs["width_m"]) / 2,
            float(specs.get("height_m", 0.85)) / 2,
        )
    if entry.type == "manipulator":
        reach = float(specs.get("reach_m", 0.5))
        return (0.15, 0.15, reach / 2)
    return (0.15, 0.15, 0.15)


def _equipment_color(eq_type: str) -> str:
    """Get RGBA color string by equipment type.

    Args:
        eq_type: Equipment type.

    Returns:
        RGBA string for MuJoCo.
    """
    colors = {
        "manipulator": "0.8 0.2 0.2 1",
        "conveyor": "0.2 0.2 0.8 1",
        "camera": "0.2 0.8 0.2 1",
        "fixture": "0.6 0.6 0.6 1",
    }
    return colors.get(eq_type, "0.5 0.5 0.5 1")


def _is_valid_mesh(mesh_path: Path) -> bool:
    """Check if mesh file exists and can be loaded by MuJoCo.

    Args:
        mesh_path: Path to mesh file.

    Returns:
        True if mesh is usable.
    """
    if not mesh_path.exists() or mesh_path.stat().st_size < 200:
        return False
    try:
        import mujoco
        test_xml = f"""<mujoco>
  <asset><mesh name="test" file="{str(mesh_path).replace(chr(92), '/')}"/></asset>
  <worldbody><geom type="mesh" mesh="test"/></worldbody>
</mujoco>"""
        mujoco.MjModel.from_xml_string(test_xml)
        return True
    except Exception:
        return False
