"""Tests for reconstruction and calibration logic."""

import numpy as np

from backend.app.models.space import Dimensions, ReferenceCalibration
from backend.app.services.reconstruction import (
    _compute_scale_factor,
    _scale_dimensions,
    check_reconstruction_deps,
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


class TestDepsCheck:
    """Tests for dependency checking."""

    def test_check_returns_dict(self) -> None:
        deps = check_reconstruction_deps()
        assert isinstance(deps, dict)
        assert "mujoco" in deps
        assert "trimesh" in deps
        assert "numpy" in deps
