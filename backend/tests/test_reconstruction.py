"""Tests for reconstruction and calibration logic."""

from pathlib import Path

import numpy as np

from backend.app.models.space import Dimensions, ReferenceCalibration
from backend.app.services.reconstruction import (
    _compute_scale_factor,
    _scale_dimensions,
    check_reconstruction_deps,
    rescale_pointcloud,
    transform_colmap_to_threejs,
)


class TestScaleCalibration:
    """Tests for scale calibration math."""

    def test_scale_factor_identity(self) -> None:
        cal = ReferenceCalibration(
            point_a=(0.0, 0.0, 0.0),
            point_b=(1.0, 0.0, 0.0),
            real_distance_m=1.0,
        )
        factor = _compute_scale_factor(cal)
        assert abs(factor - 1.0) < 1e-6

    def test_scale_factor_double(self) -> None:
        cal = ReferenceCalibration(
            point_a=(0.0, 0.0, 0.0),
            point_b=(1.0, 0.0, 0.0),
            real_distance_m=2.0,
        )
        factor = _compute_scale_factor(cal)
        assert abs(factor - 2.0) < 1e-6

    def test_scale_factor_3d_diagonal(self) -> None:
        cal = ReferenceCalibration(
            point_a=(0.0, 0.0, 0.0),
            point_b=(1.0, 1.0, 1.0),
            real_distance_m=np.sqrt(3.0),
        )
        factor = _compute_scale_factor(cal)
        assert abs(factor - 1.0) < 1e-6

    def test_scale_factor_half(self) -> None:
        cal = ReferenceCalibration(
            point_a=(0.0, 0.0, 0.0),
            point_b=(2.0, 0.0, 0.0),
            real_distance_m=1.0,
        )
        factor = _compute_scale_factor(cal)
        assert abs(factor - 0.5) < 1e-6

    def test_scale_dimensions(self) -> None:
        dims = Dimensions(width_m=10.0, length_m=8.0, ceiling_m=5.0, area_m2=80.0)
        scaled = _scale_dimensions(dims, 0.5)
        assert abs(scaled.width_m - 5.0) < 1e-6
        assert abs(scaled.length_m - 4.0) < 1e-6
        assert abs(scaled.ceiling_m - 2.5) < 1e-6
        assert abs(scaled.area_m2 - 20.0) < 1e-6


class TestCoordinateTransform:
    """Tests for COLMAP → Three.js coordinate transform."""

    def test_y_axis_flipped(self) -> None:
        points = np.array([[1.0, 2.0, 3.0]])
        result = transform_colmap_to_threejs(points)
        assert result[0, 0] == 1.0
        assert result[0, 1] == -2.0
        assert result[0, 2] == -3.0

    def test_origin_unchanged(self) -> None:
        points = np.array([[0.0, 0.0, 0.0]])
        result = transform_colmap_to_threejs(points)
        np.testing.assert_array_equal(result, points)

    def test_multiple_points(self) -> None:
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
            ]
        )
        result = transform_colmap_to_threejs(points)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(
            result,
            [[1.0, -2.0, -3.0], [-1.0, 2.0, 3.0]],
        )

    def test_preserves_x(self) -> None:
        points = np.array([[5.0, 0.0, 0.0]])
        result = transform_colmap_to_threejs(points)
        assert result[0, 0] == 5.0


class TestRescalePointcloud:
    """Tests for point cloud rescaling."""

    def test_rescale_doubles_vertices(self, tmp_path: Path) -> None:
        import trimesh

        vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cloud = trimesh.PointCloud(vertices=vertices)
        ply_path = tmp_path / "test.ply"
        cloud.export(str(ply_path))

        rescale_pointcloud(ply_path, 2.0)

        reloaded = trimesh.load(str(ply_path))
        np.testing.assert_array_almost_equal(
            reloaded.vertices,
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
        )

    def test_rescale_halves_vertices(self, tmp_path: Path) -> None:
        import trimesh

        vertices = np.array([[10.0, 20.0, 30.0]])
        cloud = trimesh.PointCloud(vertices=vertices)
        ply_path = tmp_path / "test.ply"
        cloud.export(str(ply_path))

        rescale_pointcloud(ply_path, 0.5)

        reloaded = trimesh.load(str(ply_path))
        np.testing.assert_array_almost_equal(
            reloaded.vertices,
            [[5.0, 10.0, 15.0]],
        )


class TestDepsCheck:
    """Tests for dependency checking."""

    def test_check_returns_dict(self) -> None:
        deps = check_reconstruction_deps()
        assert isinstance(deps, dict)
        assert "pycolmap" in deps
        assert "mujoco" in deps
        assert "trimesh" in deps
        assert "numpy" in deps

    def test_pycolmap_available(self) -> None:
        deps = check_reconstruction_deps()
        assert deps["pycolmap"] is True
