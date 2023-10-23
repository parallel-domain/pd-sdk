import math

import numpy as np
import pytest

from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_PD_FISHEYE
from paralleldomain.utilities.projection import (
    DistortionLookupTable,
    focal_length_to_fov,
    fov_to_focal_length,
    project_points_2d_to_3d,
    project_points_3d_to_2d,
)


@pytest.fixture
def pd_fisheye_distortion_lookup() -> DistortionLookupTable:
    theta = np.linspace(0, 3.0, 3000, endpoint=True)
    factor = np.linspace(1.0, 1.5, 3000, endpoint=True)
    theta_d = theta / factor
    distortion_lut_ndarray = np.hstack([theta_d, theta]).reshape(-1, 2)

    return DistortionLookupTable.from_ndarray(data=distortion_lut_ndarray)


@pytest.fixture
def points_3d() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [20.0, 0.0, 150.0],
            [150.0, 0.0, 20.0],
        ]
    )


@pytest.fixture
def k_matrix() -> np.ndarray:
    return np.asarray(
        [
            [1000, 0, 1920 / 2],
            [0, 800, 1080 / 2],
            [0, 0, 1],
        ]
    )


def test_opencv_pinhole_projection_shape_and_focal_point(k_matrix, points_3d):
    uv_opencv_pinhole = project_points_3d_to_2d(
        k_matrix=k_matrix,
        camera_model=CAMERA_MODEL_OPENCV_PINHOLE,
        points_3d=points_3d,
    )

    assert uv_opencv_pinhole.shape == (len(points_3d), 2)
    assert np.all(uv_opencv_pinhole[3] == k_matrix[[0, 1], [2]])


def test_opencv_fisheye_projection_shape_and_focal_point(k_matrix, points_3d):
    uv_opencv_fisheye = project_points_3d_to_2d(
        k_matrix=k_matrix,
        distortion_parameters=np.linspace(1.5, 0, 4, endpoint=False),
        camera_model=CAMERA_MODEL_OPENCV_FISHEYE,
        points_3d=points_3d,
    )

    assert uv_opencv_fisheye.shape == (len(points_3d), 2)
    assert np.all(uv_opencv_fisheye[3] == k_matrix[[0, 1], [2]])


def test_pd_fisheye_projection_shape_and_focal_point(k_matrix, pd_fisheye_distortion_lookup, points_3d):
    uv_pd_fisheye = project_points_3d_to_2d(
        k_matrix=k_matrix,
        camera_model=CAMERA_MODEL_PD_FISHEYE,
        points_3d=points_3d,
        distortion_lookup=pd_fisheye_distortion_lookup,
    )

    assert uv_pd_fisheye.shape == (len(points_3d), 2)
    assert np.all(uv_pd_fisheye[3] == k_matrix[[0, 1], [2]])


def test_opencv_fisheye_3d_to_2d_and_back(k_matrix: np.ndarray):
    d = np.linspace(1.5, 0, 4, endpoint=False)
    points_3d = np.asarray(
        [
            [1.0, 0.0, 10.0],
            [1.0, 5.0, 10.0],
            [2.0, 1.0, 10.0],
            [2.0, 0.0, 10.0],
            [0.20, -1.23, 10.0],
            [5.0, -2.2, 10.0],
        ]
    )
    uv_opencv_fisheye = project_points_3d_to_2d(
        k_matrix=k_matrix,
        distortion_parameters=d,
        camera_model=CAMERA_MODEL_OPENCV_FISHEYE,
        points_3d=points_3d,
    )
    result = project_points_2d_to_3d(
        k_matrix=k_matrix,
        camera_model=CAMERA_MODEL_OPENCV_FISHEYE,
        points_2d=uv_opencv_fisheye[:, :2],
        depth=10 * np.ones((1920, 1080, 1)),
        distortion_parameters=d,
    )

    assert np.allclose(result, points_3d)


class TestFovFocalFunctions:
    def test_fov_to_focal_length_positive(self):
        # Test with positive fov
        math.isclose(fov_to_focal_length(math.pi / 4, 1000), 707.10678118, abs_tol=10e-5)

    def test_fov_to_focal_length_zero(self):
        # Test with fov as zero
        assert fov_to_focal_length(0.0, 1000) == 0.0

    def test_fov_to_focal_length_negative(self):
        # Test with negative fov
        assert fov_to_focal_length(-math.pi / 4, 1000) == 0.0

    def test_focal_length_to_fov_positive(self):
        # Test with positive focal_length
        assert math.isclose(focal_length_to_fov(1000, 1000), 0.92729, abs_tol=10e-5)

    def test_focal_length_to_fov_zero(self):
        # Test with focal_length as zero
        assert focal_length_to_fov(0.0, 1000) == 0.0

    def test_focal_length_to_fov_negative(self):
        # Test with negative focal_length
        assert focal_length_to_fov(-707.10678118, 1000) == 0.0

    def test_fov_to_focal_length_inverse(self):
        # Test that converting from fov to focal_length and back gives the original fov
        original_fov = math.pi / 3
        length = 1000
        calculated_focal_length = fov_to_focal_length(original_fov, length)
        calculated_fov = focal_length_to_fov(calculated_focal_length, length)
        assert math.isclose(original_fov, calculated_fov, abs_tol=10e-5)

    def test_focal_length_to_fov_inverse(self):
        # Test that converting from focal_length to fov and back gives the original focal_length
        original_focal_length = 500.0
        length = 1000
        calculated_fov = focal_length_to_fov(original_focal_length, length)
        calculated_focal_length = fov_to_focal_length(calculated_fov, length)
        assert math.isclose(original_focal_length, calculated_focal_length, abs_tol=10e-5)
