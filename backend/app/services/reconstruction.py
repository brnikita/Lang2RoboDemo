"""DISCOVERSE Real2Sim reconstruction pipeline.

Orchestrates: photos → COLMAP SfM → 3D Gaussian Splatting → mesh → MJCF export.
Requires external tools: COLMAP, DISCOVERSE, optionally Blender.
"""

import asyncio
import logging
import shutil
import subprocess
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
    deps: dict[str, bool] = {}
    deps["colmap"] = shutil.which("colmap") is not None
    deps["mujoco"] = _check_python_module("mujoco")
    deps["trimesh"] = _check_python_module("trimesh")
    deps["numpy"] = _check_python_module("numpy")
    return deps


async def reconstruct_scene(
    photos_dir: Path,
    output_dir: Path,
) -> SceneReconstruction:
    """Run full Real2Sim reconstruction pipeline.

    Args:
        photos_dir: Directory containing room photos (10-30 images).
        output_dir: Output directory for reconstruction artifacts.

    Returns:
        SceneReconstruction with paths to mesh, MJCF, and point cloud.

    Raises:
        FileNotFoundError: If photos_dir doesn't exist or is empty.
        RuntimeError: If reconstruction pipeline fails.
    """
    _validate_photos_dir(photos_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = output_dir / "sparse"
    dense_dir = output_dir / "dense"
    mesh_path = output_dir / "mesh.obj"
    pointcloud_path = output_dir / "pointcloud.ply"
    mjcf_path = output_dir / "scene.xml"

    await _run_colmap_sfm(photos_dir, sparse_dir)
    await _run_colmap_dense(photos_dir, sparse_dir, dense_dir)
    await _extract_mesh(dense_dir, mesh_path, pointcloud_path)
    _generate_base_mjcf(mesh_path, mjcf_path)

    dimensions = _estimate_dimensions_from_mesh(mesh_path)

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
    """Apply scale calibration to reconstruction using a known measurement.

    Args:
        reconstruction: Uncalibrated reconstruction.
        calibration: Two points + real-world distance from user.

    Returns:
        New SceneReconstruction with calibrated dimensions.
    """
    scale_factor = _compute_scale_factor(calibration)
    scaled_dims = _scale_dimensions(reconstruction.dimensions, scale_factor)
    _apply_scale_to_mjcf(reconstruction.mjcf_path, scale_factor)
    _apply_scale_to_mesh(reconstruction.mesh_path, scale_factor)

    return SceneReconstruction(
        mesh_path=reconstruction.mesh_path,
        mjcf_path=reconstruction.mjcf_path,
        pointcloud_path=reconstruction.pointcloud_path,
        dimensions=scaled_dims,
    )


def _validate_photos_dir(photos_dir: Path) -> None:
    """Validate that photo directory exists and contains images.

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


async def _run_colmap_sfm(
    photos_dir: Path, sparse_dir: Path,
) -> None:
    """Run COLMAP Structure-from-Motion.

    Args:
        photos_dir: Directory with input photos.
        sparse_dir: Output directory for sparse reconstruction.

    Raises:
        RuntimeError: If COLMAP fails.
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)
    db_path = sparse_dir.parent / "database.db"

    await _run_subprocess([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(photos_dir),
    ])
    await _run_subprocess([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
    ])
    await _run_subprocess([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(photos_dir),
        "--output_path", str(sparse_dir),
    ])


async def _run_colmap_dense(
    photos_dir: Path, sparse_dir: Path, dense_dir: Path,
) -> None:
    """Run COLMAP dense reconstruction.

    Args:
        photos_dir: Directory with input photos.
        sparse_dir: Sparse reconstruction output.
        dense_dir: Output directory for dense reconstruction.

    Raises:
        RuntimeError: If COLMAP fails.
    """
    dense_dir.mkdir(parents=True, exist_ok=True)

    await _run_subprocess([
        "colmap", "image_undistorter",
        "--image_path", str(photos_dir),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(dense_dir),
    ])
    await _run_subprocess([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
    ])
    await _run_subprocess([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(dense_dir / "fused.ply"),
    ])


async def _extract_mesh(
    dense_dir: Path, mesh_path: Path, pointcloud_path: Path,
) -> None:
    """Extract mesh from dense reconstruction using Poisson surface.

    Args:
        dense_dir: Dense reconstruction directory.
        mesh_path: Output mesh file path.
        pointcloud_path: Output point cloud file path.
    """
    import trimesh

    fused_ply = dense_dir / "fused.ply"
    if fused_ply.exists():
        shutil.copy2(fused_ply, pointcloud_path)

    await _run_subprocess([
        "colmap", "poisson_mesher",
        "--input_path", str(dense_dir / "fused.ply"),
        "--output_path", str(mesh_path),
    ])

    if not mesh_path.exists() and pointcloud_path.exists():
        cloud = trimesh.load(str(pointcloud_path))
        if hasattr(cloud, "convex_hull"):
            cloud.convex_hull.export(str(mesh_path))


def _generate_base_mjcf(mesh_path: Path, mjcf_path: Path) -> None:
    """Generate a base MJCF scene file referencing the room mesh.

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


def _estimate_dimensions_from_mesh(mesh_path: Path) -> Dimensions:
    """Estimate room dimensions from mesh bounding box.

    Args:
        mesh_path: Path to mesh file.

    Returns:
        Estimated dimensions (uncalibrated — arbitrary scale).
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    bounds = mesh.bounding_box.extents
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


async def _run_subprocess(cmd: list[str]) -> str:
    """Run external command asynchronously.

    Args:
        cmd: Command and arguments.

    Returns:
        Combined stdout + stderr output.

    Raises:
        RuntimeError: If command fails.
    """
    logger.info("Running: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode() + stderr.decode()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (code {proc.returncode}): "
            f"{' '.join(cmd)}\n{output}"
        )
    return output


def _check_python_module(name: str) -> bool:
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
