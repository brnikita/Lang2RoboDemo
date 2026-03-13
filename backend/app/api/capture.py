"""Capture API — photo upload, reconstruction, and scene analysis."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.core.claude import get_claude_client
from backend.app.core.config import get_settings
from backend.app.models.space import ReferenceCalibration, SpaceModel
from backend.app.services.reconstruction import (
    calibrate_scale,
    reconstruct_scene,
)
from backend.app.services.vision import analyze_scene, build_space_model

__all__ = ["router"]

router = APIRouter(prefix="/api/capture", tags=["capture"])


@router.post("")
async def upload_photos(
    photos: list[UploadFile] = File(...),
) -> dict:
    """Upload room photos and start reconstruction.

    Args:
        photos: Room photos (10-30 images).

    Returns:
        Project ID and reconstruction status.
    """
    if len(photos) < 3:
        raise HTTPException(400, "At least 3 photos required")

    project_id = str(uuid.uuid4())
    project_dir = _get_project_dir(project_id)
    photos_dir = project_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    for photo in photos:
        dest = photos_dir / photo.filename
        content = await photo.read()
        dest.write_bytes(content)

    output_dir = project_dir / "reconstruction"
    reconstruction = await reconstruct_scene(photos_dir, output_dir)

    _save_reconstruction_meta(project_dir, reconstruction)

    return {
        "project_id": project_id,
        "status": "reconstructed",
        "dimensions": reconstruction.dimensions.model_dump(),
    }


@router.post("/{project_id}/calibrate")
async def calibrate_and_analyze(
    project_id: str,
    calibration: ReferenceCalibration,
) -> SpaceModel:
    """Apply scale calibration and run Claude Vision analysis.

    Args:
        project_id: Project identifier from upload.
        calibration: Two points + real distance for scale.

    Returns:
        Complete SpaceModel with zones, equipment, doors, windows.
    """
    project_dir = _get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(404, f"Project {project_id} not found")

    reconstruction = _load_reconstruction_meta(project_dir)
    calibrated = calibrate_scale(reconstruction, calibration)

    photos_dir = project_dir / "photos"
    photo_files = _list_photos(photos_dir)

    client = get_claude_client()
    analysis = await analyze_scene(client, photo_files, calibrated)

    space_model = build_space_model(calibrated, analysis)

    space_path = project_dir / "space_model.json"
    space_path.write_text(
        space_model.model_dump_json(indent=2), encoding="utf-8",
    )

    return space_model


@router.get("/{project_id}/pointcloud")
async def get_pointcloud(project_id: str) -> FileResponse:
    """Serve the reconstructed point cloud PLY file.

    Args:
        project_id: Project identifier.

    Returns:
        PLY file response.
    """
    project_dir = _get_project_dir(project_id)
    ply_path = project_dir / "reconstruction" / "pointcloud.ply"
    if not ply_path.exists() or ply_path.stat().st_size == 0:
        raise HTTPException(404, "Point cloud not available")
    return FileResponse(ply_path, media_type="application/octet-stream")


@router.get("/{project_id}/mesh")
async def get_mesh(project_id: str) -> FileResponse:
    """Serve the reconstructed mesh OBJ file.

    Args:
        project_id: Project identifier.

    Returns:
        OBJ file response.
    """
    project_dir = _get_project_dir(project_id)
    mesh_path = project_dir / "reconstruction" / "mesh.obj"
    if not mesh_path.exists() or mesh_path.stat().st_size == 0:
        raise HTTPException(404, "Mesh not available")
    return FileResponse(mesh_path, media_type="application/octet-stream")


def _get_project_dir(project_id: str) -> Path:
    """Get project data directory path.

    Args:
        project_id: Project identifier.

    Returns:
        Path to project directory.
    """
    settings = get_settings()
    return settings.DATA_DIR / "projects" / project_id


def _list_photos(photos_dir: Path) -> list[Path]:
    """List all image files in a directory.

    Args:
        photos_dir: Directory to scan.

    Returns:
        Sorted list of image file paths.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(
        f for f in photos_dir.iterdir()
        if f.suffix.lower() in image_exts
    )


def _save_reconstruction_meta(
    project_dir: Path,
    reconstruction: "SceneReconstruction",
) -> None:
    """Save reconstruction metadata to project directory.

    Args:
        project_dir: Project directory.
        reconstruction: Reconstruction data to save.
    """
    from backend.app.models.space import SceneReconstruction

    meta_path = project_dir / "reconstruction_meta.json"
    meta_path.write_text(
        reconstruction.model_dump_json(indent=2), encoding="utf-8",
    )


def _load_reconstruction_meta(project_dir: Path) -> "SceneReconstruction":
    """Load reconstruction metadata from project directory.

    Args:
        project_dir: Project directory.

    Returns:
        SceneReconstruction instance.

    Raises:
        HTTPException: If metadata file not found.
    """
    from backend.app.models.space import SceneReconstruction

    meta_path = project_dir / "reconstruction_meta.json"
    if not meta_path.exists():
        raise HTTPException(
            404, f"Reconstruction not found in project {project_dir.name}"
        )
    return SceneReconstruction.model_validate_json(
        meta_path.read_text(encoding="utf-8")
    )
