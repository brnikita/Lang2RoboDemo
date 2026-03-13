"""Shared test fixtures for Lang2Robo backend tests."""

from pathlib import Path

import pytest

from backend.app.models.space import (
    Dimensions,
    SceneReconstruction,
    SpaceModel,
)


@pytest.fixture
def sample_dimensions() -> Dimensions:
    """Create sample room dimensions for testing."""
    return Dimensions(width_m=6.0, length_m=5.0, ceiling_m=3.0, area_m2=30.0)


@pytest.fixture
def sample_reconstruction(
    tmp_path: Path, sample_dimensions: Dimensions
) -> SceneReconstruction:
    """Create sample scene reconstruction for testing."""
    mesh = tmp_path / "mesh.obj"
    mjcf = tmp_path / "scene.xml"
    pc = tmp_path / "pointcloud.ply"
    mesh.touch()
    mjcf.touch()
    pc.touch()
    return SceneReconstruction(
        mesh_path=mesh,
        mjcf_path=mjcf,
        pointcloud_path=pc,
        dimensions=sample_dimensions,
    )


@pytest.fixture
def sample_space_model(
    sample_dimensions: Dimensions,
    sample_reconstruction: SceneReconstruction,
) -> SpaceModel:
    """Create a minimal SpaceModel for testing."""
    return SpaceModel(
        dimensions=sample_dimensions,
        reconstruction=sample_reconstruction,
    )
