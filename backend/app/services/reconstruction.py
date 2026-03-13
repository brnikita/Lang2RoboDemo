"""Real2Sim reconstruction pipeline via pycolmap.

Orchestrates: photos → feature extraction → matching → SfM → dense → mesh → MJCF.
Uses pycolmap Python API — no CLI dependency.
"""

import asyncio
import logging
import shutil
from pathlib import Path

import numpy as np

from backend.app.models.space import Dimensions, ReferenceCalibration, SceneReconstruction

__all__ = [
    "reconstruct_scene",
    "calibrate_scale",
    "check_reconstruction_deps",
]

logger = logging.getLogger(__name__)


def check_reconstruction_deps() -> dict[str, bool]:
    """Check which reconstruction dependencies are available.

    Returns:
        Mapping of dependency name to availability.
    """
    return {
        "pycolmap": _check_module("pycolmap"),
        "mujoco": _check_module("mujoco"),
        "trimesh": _check_module("trimesh"),
        "numpy": _check_module("numpy"),
    }


async def reconstruct_scene(
    photos_dir: Path,
    output_dir: Path,
) -> SceneReconstruction:
    """Run reconstruction: photos → sparse SfM → point cloud → mesh → MJCF.

    Args:
        photos_dir: Directory containing room photos (3+ images).
        output_dir: Output directory for reconstruction artifacts.

    Returns:
        SceneReconstruction with paths to mesh, MJCF, and point cloud.

    Raises:
        FileNotFoundError: If photos_dir is empty.
        RuntimeError: If reconstruction fails.
    """
    _validate_photos_dir(photos_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    mesh_path = output_dir / "mesh.obj"
    pointcloud_path = output_dir / "pointcloud.ply"
    mjcf_path = output_dir / "scene.xml"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _run_pycolmap_pipeline,
        photos_dir, db_path, sparse_dir, pointcloud_path, mesh_path,
    )

    _generate_base_mjcf(mesh_path, mjcf_path)
    dimensions = _estimate_dimensions(mesh_path, pointcloud_path)

    return SceneReconstruction(
        mesh_path=mesh_path,
        mjcf_path=mjcf_path,
        pointcloud_path=pointcloud_path,
        dimensions=dimensions,
    )


def calibrate_scale(
    reconstruction: SceneReconstruction,
    calibration: ReferenceCalibration,
) -> SceneReconstruction:
    """Apply scale calibration using a known real-world measurement.

    Args:
        reconstruction: Uncalibrated reconstruction.
        calibration: Two points + real-world distance from user.

    Returns:
        New SceneReconstruction with calibrated dimensions.
    """
    scale_factor = _compute_scale_factor(calibration)
    scaled_dims = _scale_dimensions(
        reconstruction.dimensions, scale_factor,
    )
    _apply_scale_to_mjcf(reconstruction.mjcf_path, scale_factor)
    if reconstruction.mesh_path.stat().st_size > 0:
        _apply_scale_to_mesh(reconstruction.mesh_path, scale_factor)

    return SceneReconstruction(
        mesh_path=reconstruction.mesh_path,
        mjcf_path=reconstruction.mjcf_path,
        pointcloud_path=reconstruction.pointcloud_path,
        dimensions=scaled_dims,
    )


def _run_pycolmap_pipeline(
    photos_dir: Path,
    db_path: Path,
    sparse_dir: Path,
    pointcloud_path: Path,
    mesh_path: Path,
) -> None:
    """Execute full pycolmap reconstruction pipeline (blocking).

    Args:
        photos_dir: Input photos directory.
        db_path: COLMAP database path.
        sparse_dir: Sparse reconstruction output directory.
        pointcloud_path: Output fused point cloud.
        mesh_path: Output mesh file.
    """
    import pycolmap

    sparse_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting features from %s", photos_dir)

    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.max_num_features = 8192
    pycolmap.extract_features(db_path, photos_dir, sift_options=sift_opts)

    logger.info("Matching features exhaustively")
    match_opts = pycolmap.SiftMatchingOptions()
    match_opts.max_ratio = 0.9
    match_opts.max_distance = 0.9
    pycolmap.match_exhaustive(db_path, sift_options=match_opts)

    logger.info("Running incremental SfM")
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_num_matches = 10
    reconstructions = pycolmap.incremental_mapping(
        db_path, photos_dir, sparse_dir, options=mapper_opts,
    )

    if not reconstructions:
        raise RuntimeError(
            "SfM failed — no reconstruction produced. "
            "Ensure photos have sufficient overlap."
        )

    best = reconstructions[0]
    logger.info(
        "SfM complete: %d images, %d points",
        best.num_reg_images(), best.num_points3D(),
    )

    _export_pointcloud(best, pointcloud_path)
    _pointcloud_to_mesh(pointcloud_path, mesh_path)


def _export_pointcloud(
    reconstruction, pointcloud_path: Path,
) -> None:
    """Export 3D points from reconstruction as PLY.

    Args:
        reconstruction: pycolmap Reconstruction object.
        pointcloud_path: Output PLY file path.
    """
    import trimesh

    points = []
    colors = []
    for point3d in reconstruction.points3D.values():
        points.append(point3d.xyz)
        colors.append(point3d.color)

    if not points:
        raise RuntimeError("No 3D points in reconstruction")

    cloud = trimesh.PointCloud(
        vertices=np.array(points),
        colors=np.array(colors, dtype=np.uint8),
    )
    cloud.export(str(pointcloud_path))
    logger.info("Exported %d points to %s", len(points), pointcloud_path)


def _pointcloud_to_mesh(
    pointcloud_path: Path, mesh_path: Path,
) -> None:
    """Convert point cloud to mesh via convex hull or ball-pivoting.

    Args:
        pointcloud_path: Input PLY point cloud.
        mesh_path: Output mesh file.
    """
    import trimesh

    cloud = trimesh.load(str(pointcloud_path))
    if hasattr(cloud, "convex_hull"):
        hull = cloud.convex_hull
        hull.export(str(mesh_path))
        logger.info("Mesh exported: %d faces", len(hull.faces))
    else:
        logger.warning("Could not generate mesh from point cloud")
        mesh_path.touch()


def _validate_photos_dir(photos_dir: Path) -> None:
    """Validate photo directory exists and has images.

    Args:
        photos_dir: Directory to validate.

    Raises:
        FileNotFoundError: If invalid.
    """
    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    photos = [
        f for f in photos_dir.iterdir()
        if f.suffix.lower() in image_exts
    ]
    if len(photos) < 3:
        raise FileNotFoundError(
            f"Need at least 3 photos, found {len(photos)} in {photos_dir}"
        )


def _generate_base_mjcf(mesh_path: Path, mjcf_path: Path) -> None:
    """Generate base MJCF scene file referencing the room mesh.

    Args:
        mesh_path: Path to the room mesh.
        mjcf_path: Output MJCF file path.
    """
    rel_mesh = mesh_path.name
    mjcf_content = f"""<mujoco model="reconstructed_scene">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <mesh name="room_mesh" file="{rel_mesh}" scale="1 1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="10 10 0.01" rgba="0.8 0.8 0.8 1"/>

    <body name="room" pos="0 0 0">
      <geom name="room_visual" type="mesh" mesh="room_mesh"
            contype="1" conaffinity="1" rgba="0.9 0.9 0.9 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""
    mjcf_path.write_text(mjcf_content, encoding="utf-8")


def _estimate_dimensions(
    mesh_path: Path, pointcloud_path: Path,
) -> Dimensions:
    """Estimate room dimensions from mesh or point cloud bounding box.

    Args:
        mesh_path: Path to mesh file.
        pointcloud_path: Path to point cloud (fallback).

    Returns:
        Estimated dimensions (uncalibrated scale).
    """
    import trimesh

    src = mesh_path if mesh_path.stat().st_size > 0 else pointcloud_path
    geom = trimesh.load(str(src), force="mesh")
    bounds = geom.bounding_box.extents

    return Dimensions(
        width_m=float(bounds[0]),
        length_m=float(bounds[1]),
        ceiling_m=float(bounds[2]),
        area_m2=float(bounds[0] * bounds[1]),
    )


def _compute_scale_factor(calibration: ReferenceCalibration) -> float:
    """Compute scale factor from calibration measurement.

    Args:
        calibration: Two points + known real distance.

    Returns:
        Scale factor to multiply mesh coordinates by.
    """
    a = np.array(calibration.point_a)
    b = np.array(calibration.point_b)
    mesh_distance = float(np.linalg.norm(a - b))
    if mesh_distance < 1e-6:
        raise ValueError("Calibration points are too close together")
    return calibration.real_distance_m / mesh_distance


def _scale_dimensions(dims: Dimensions, scale: float) -> Dimensions:
    """Apply scale factor to dimensions.

    Args:
        dims: Original dimensions.
        scale: Scale factor.

    Returns:
        Scaled dimensions.
    """
    return Dimensions(
        width_m=dims.width_m * scale,
        length_m=dims.length_m * scale,
        ceiling_m=dims.ceiling_m * scale,
        area_m2=dims.width_m * scale * dims.length_m * scale,
    )


def _apply_scale_to_mjcf(mjcf_path: Path, scale: float) -> None:
    """Update MJCF mesh scale attribute.

    Args:
        mjcf_path: Path to MJCF file.
        scale: Scale factor.
    """
    content = mjcf_path.read_text(encoding="utf-8")
    content = content.replace(
        'scale="1 1 1"',
        f'scale="{scale:.6f} {scale:.6f} {scale:.6f}"',
    )
    mjcf_path.write_text(content, encoding="utf-8")


def _apply_scale_to_mesh(mesh_path: Path, scale: float) -> None:
    """Scale mesh geometry in-place.

    Args:
        mesh_path: Path to mesh file.
        scale: Scale factor.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(scale)
    mesh.export(str(mesh_path))


def _check_module(name: str) -> bool:
    """Check if a Python module is importable.

    Args:
        name: Module name.

    Returns:
        True if importable.
    """
    try:
        __import__(name)
        return True
    except ImportError:
        return False
