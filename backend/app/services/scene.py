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
from backend.app.models.space import Dimensions, SpaceModel
from backend.app.services.downloader import find_mjcf_in_dir

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
        root,
        worldbody,
        recommendation.equipment,
        model_dirs,
        catalog,
        output_path.parent,
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
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "width": "512",
            "height": "512",
            "rgb1": "0.8 0.8 0.8",
            "rgb2": "0.6 0.6 0.6",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "grid_mat",
            "texture": "grid",
            "texrepeat": "4 4",
            "reflectance": "0.1",
        },
    )


def _include_room_mesh(
    asset: ET.Element,
    space: SpaceModel,
) -> None:
    """Include room mesh from DISCOVERSE reconstruction.

    Args:
        asset: Asset XML element.
        space: Room model with reconstruction paths.
    """
    mesh_path = space.reconstruction.mesh_path
    if mesh_path.exists() and _is_valid_mesh(mesh_path):
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "room_mesh",
                "file": str(mesh_path).replace("\\", "/"),
            },
        )


def _add_lighting(
    worldbody: ET.Element,
    dims: Dimensions,
) -> None:
    """Add scene lighting.

    Args:
        worldbody: Worldbody XML element.
        dims: Room dimensions for positioning.
    """
    cx = dims.width_m / 2
    cy = dims.length_m / 2
    ET.SubElement(
        worldbody,
        "light",
        {
            "pos": f"{cx:.2f} {cy:.2f} {dims.ceiling_m:.2f}",
            "dir": "0 0 -1",
            "diffuse": "1 1 1",
        },
    )


def _add_floor(
    worldbody: ET.Element,
    dims: Dimensions,
) -> None:
    """Add floor plane.

    Args:
        worldbody: Worldbody XML element.
        dims: Room dimensions for sizing.
    """
    sx = dims.width_m / 2
    sy = dims.length_m / 2
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": f"{sx:.2f} {sy:.2f} 0.01",
            "material": "grid_mat",
        },
    )


def _add_room_body(
    worldbody: ET.Element,
    space: SpaceModel,
) -> None:
    """Add room mesh as visual body.

    Args:
        worldbody: Worldbody XML element.
        space: Room model.
    """
    mesh_path = space.reconstruction.mesh_path
    if mesh_path.exists() and _is_valid_mesh(mesh_path):
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "room",
                "pos": "0 0 0",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": "room_visual",
                "type": "mesh",
                "mesh": "room_mesh",
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0.9 0.9 0.9 0.3",
            },
        )


def _add_existing_equipment(
    worldbody: ET.Element,
    space: SpaceModel,
) -> None:
    """Add existing equipment as static bodies.

    Args:
        worldbody: Worldbody XML element.
        space: Room model with existing equipment.
    """
    for eq in space.existing_equipment:
        pos = f"{eq.position[0]:.3f} {eq.position[1]:.3f} {eq.position[2]:.3f}"
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": eq.name,
                "pos": pos,
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{eq.name}_geom",
                "type": "box",
                "size": "0.2 0.2 0.4",
                "rgba": "0.6 0.4 0.2 1",
            },
        )


def _add_new_equipment(
    root: ET.Element,
    worldbody: ET.Element,
    placements: list[EquipmentPlacement],
    model_dirs: dict[str, Path],
    catalog: dict[str, EquipmentEntry],
    scene_dir: Path,
) -> None:
    """Add new equipment from recommendation, dispatching by type.

    Args:
        root: Root mujoco element (for includes).
        worldbody: Worldbody XML element.
        placements: Equipment placements.
        model_dirs: Model directory mapping.
        catalog: Equipment catalog.
        scene_dir: Scene directory for relative paths.
    """
    name_counts: dict[str, int] = {}
    for placement in placements:
        entry = catalog.get(placement.equipment_id)
        if not entry:
            continue

        count = name_counts.get(placement.equipment_id, 0)
        name_counts[placement.equipment_id] = count + 1
        body_name = placement.equipment_id if count == 0 else f"{placement.equipment_id}_{count}"

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


def _add_manipulator_to_scene(
    root: ET.Element,
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    body_name: str,
    model_dirs: dict[str, Path],
    _scene_dir: Path,
) -> None:
    """Inline real MJCF model for a manipulator.

    Parses the robot MJCF file, merges top-level sections (compiler,
    default, asset, actuator, etc.) into the scene root, and places
    the robot's body tree inside a positioning wrapper body.

    Args:
        root: Root mujoco element for merging top-level sections.
        worldbody: Worldbody XML element.
        placement: Equipment placement data.
        body_name: Unique body name.
        model_dirs: Model directory mapping.
        scene_dir: Scene directory for relative paths.
    """
    model_dir = model_dirs.get(placement.equipment_id)
    mjcf_file = find_mjcf_in_dir(model_dir) if model_dir else None

    if mjcf_file:
        _inline_robot_model(
            root,
            worldbody,
            mjcf_file,
            placement,
            body_name,
        )
    else:
        _add_fallback_box(worldbody, placement, body_name)


def _inline_robot_model(
    root: ET.Element,
    worldbody: ET.Element,
    mjcf_file: Path,
    placement: EquipmentPlacement,
    body_name: str,
) -> None:
    """Parse robot MJCF and inline it into the scene.

    Merges top-level sections into root and places body tree
    inside a wrapper body at the desired position.

    Args:
        root: Root mujoco element.
        worldbody: Worldbody XML element.
        mjcf_file: Path to robot MJCF file.
        placement: Equipment placement data.
        body_name: Unique body name.
    """
    robot_tree = ET.parse(str(mjcf_file))
    robot_root = robot_tree.getroot()

    _resolve_meshdir(robot_root, mjcf_file.parent)
    _merge_top_level_sections(root, robot_root)

    pos = _format_pos(placement.position)
    euler = f"0 0 {math.radians(placement.orientation_deg):.4f}"
    wrapper = ET.SubElement(
        worldbody,
        "body",
        {"name": body_name, "pos": pos, "euler": euler},
    )

    robot_wb = robot_root.find("worldbody")
    if robot_wb is not None:
        for child in robot_wb:
            if child.tag == "light":
                continue
            wrapper.append(child)


def _resolve_meshdir(
    robot_root: ET.Element,
    model_dir: Path,
) -> None:
    """Resolve meshdir so mesh paths are absolute.

    Updates mesh file attributes in asset to use absolute paths
    based on the compiler meshdir setting.

    Args:
        robot_root: Robot MJCF root element.
        model_dir: Directory containing the robot MJCF file.
    """
    compiler = robot_root.find("compiler")
    meshdir = model_dir.resolve()
    if compiler is not None:
        raw = compiler.get("meshdir")
        if raw:
            meshdir = (model_dir / raw).resolve()
    asset = robot_root.find("asset")
    if asset is None:
        return
    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if file_attr and not Path(file_attr).is_absolute():
            abs_path = str(meshdir / file_attr).replace("\\", "/")
            mesh.set("file", abs_path)


_MERGEABLE_SECTIONS = [
    "compiler",
    "option",
    "size",
    "default",
    "asset",
    "tendon",
    "equality",
    "actuator",
    "sensor",
    "keyframe",
    "contact",
]


def _merge_top_level_sections(
    scene_root: ET.Element,
    robot_root: ET.Element,
) -> None:
    """Merge robot top-level MJCF sections into the scene root.

    For container sections (asset, actuator, etc.), children are
    appended. For singleton sections (compiler, option), they are
    added only if not already present.

    Args:
        scene_root: Scene mujoco root element.
        robot_root: Robot mujoco root element.
    """
    singleton_tags = {"compiler", "option", "size"}
    for tag in _MERGEABLE_SECTIONS:
        robot_section = robot_root.find(tag)
        if robot_section is None:
            continue
        scene_section = scene_root.find(tag)
        if tag in singleton_tags:
            if scene_section is None:
                scene_root.insert(0, robot_section)
        else:
            if scene_section is None:
                scene_root.append(robot_section)
            else:
                for child in robot_section:
                    scene_section.append(child)


def _add_conveyor_to_scene(
    root: ET.Element,
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    body_name: str,
    entry: EquipmentEntry,
) -> None:
    """Add parametric conveyor with belt surface, rollers, and frame.

    Args:
        root: Root mujoco element.
        worldbody: Worldbody XML element.
        placement: Equipment placement data.
        body_name: Unique body name.
        entry: Equipment catalog entry.
    """
    specs = entry.specs
    length = float(specs.get("length_m", 0.5))
    width = float(specs.get("width_m", 0.15))
    height = float(specs.get("height_m", 0.85))

    pos = _format_pos(placement.position)
    euler = f"0 0 {math.radians(placement.orientation_deg):.4f}"
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": body_name, "pos": pos, "euler": euler},
    )

    half_l = length / 2
    half_w = width / 2

    # Belt surface (high friction)
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_belt",
            "type": "box",
            "size": f"{half_l:.3f} {half_w:.3f} 0.005",
            "pos": f"0 0 {height:.3f}",
            "friction": "1 0.005 0.0001",
            "rgba": "0.3 0.3 0.3 1",
        },
    )

    # Support frame
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_frame",
            "type": "box",
            "size": f"{half_l:.3f} {half_w + 0.01:.3f} {height / 2:.3f}",
            "pos": f"0 0 {height / 2:.3f}",
            "rgba": "0.4 0.4 0.5 1",
        },
    )

    # End rollers
    for side, x_off in [("left", -half_l), ("right", half_l)]:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{body_name}_roller_{side}",
                "type": "cylinder",
                "size": f"0.02 {half_w:.3f}",
                "pos": f"{x_off:.3f} 0 {height:.3f}",
                "euler": "1.5708 0 0",
                "rgba": "0.5 0.5 0.5 1",
            },
        )

    # Ensure actuator section exists (for Phase 4 belt velocity)
    actuator = root.find("actuator")
    if actuator is None:
        ET.SubElement(root, "actuator")


def _add_camera_body_to_scene(
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    body_name: str,
    entry: EquipmentEntry,
) -> None:
    """Add camera housing body.

    Args:
        worldbody: Worldbody XML element.
        placement: Equipment placement data.
        body_name: Unique body name.
        entry: Equipment catalog entry.
    """
    height = float(entry.specs.get("mounting_height_m", 1.5))
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": body_name,
            "pos": f"{placement.position[0]:.3f} {placement.position[1]:.3f} {height:.3f}",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_housing",
            "type": "box",
            "size": "0.03 0.03 0.02",
            "rgba": "0.15 0.15 0.15 1",
        },
    )


def _add_fixture_to_scene(
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    body_name: str,
    entry: EquipmentEntry,
) -> None:
    """Add fixture as static box geom.

    Args:
        worldbody: Worldbody XML element.
        placement: Equipment placement data.
        body_name: Unique body name.
        entry: Equipment catalog entry.
    """
    pos = _format_pos(placement.position)
    euler = f"0 0 {math.radians(placement.orientation_deg):.4f}"
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": body_name, "pos": pos, "euler": euler},
    )
    size = _equipment_half_size(entry)
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_geom",
            "type": "box",
            "size": f"{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}",
            "rgba": _equipment_color(entry.type),
        },
    )


def _add_fallback_box(
    worldbody: ET.Element,
    placement: EquipmentPlacement,
    body_name: str,
) -> None:
    """Add a generic fallback box when no model is available.

    Args:
        worldbody: Worldbody XML element.
        placement: Equipment placement data.
        body_name: Unique body name.
    """
    pos = _format_pos(placement.position)
    euler = f"0 0 {math.radians(placement.orientation_deg):.4f}"
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": body_name, "pos": pos, "euler": euler},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_geom",
            "type": "box",
            "size": "0.15 0.15 0.15",
            "rgba": "0.8 0.2 0.2 1",
        },
    )


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

            body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": name,
                    "pos": pos,
                },
            )
            ET.SubElement(body, "freejoint", {"name": f"{name}_joint"})

            geom_attrs = {
                "name": f"{name}_geom",
                "type": obj.shape,
                "mass": f"{obj.mass_kg:.4f}",
                "rgba": "0.2 0.6 0.2 1",
            }
            if obj.shape == "box":
                geom_attrs["size"] = (
                    f"{obj.size[0] / 2:.4f} {obj.size[1] / 2:.4f} {obj.size[2] / 2:.4f}"
                )
            elif obj.shape == "cylinder":
                geom_attrs["size"] = f"{obj.size[0]:.4f} {obj.size[1] / 2:.4f}"
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
        pos = f"{placement.position[0]:.3f} {placement.position[1]:.3f} {float(height):.3f}"
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": placement.equipment_id,
                "pos": pos,
                "zaxis": "0 0 -1",
                "fovy": str(int(fov)),
            },
        )


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
    _root: ET.Element,
    body: ET.Element,
    mjcf_file: Path,
    _scene_dir: Path,
) -> None:
    """Include external MJCF model into scene.

    Args:
        root: Root mujoco element.
        body: Parent body element.
        mjcf_file: Path to external MJCF.
        scene_dir: Scene directory for relative paths.
    """
    body.set("childclass", mjcf_file.stem)
    ET.SubElement(
        body,
        "include",
        {
            "file": str(mjcf_file),
        },
    )


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
    if not mesh_path.exists():
        return False
    if mesh_path.stat().st_size < 10000:
        return False
    try:
        import mujoco

        path_str = str(mesh_path).replace(chr(92), "/")
        test_xml = (
            f'<mujoco><asset><mesh name="t" file="{path_str}"/></asset>'
            f'<worldbody><geom type="mesh" mesh="t"/></worldbody></mujoco>'
        )
        mujoco.MjModel.from_xml_string(test_xml)
        return True
    except Exception:
        return False
