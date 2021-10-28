import numpy as np
import pytest

from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_PD_FISHEYE
from paralleldomain.utilities.projection import DistortionLookupTable, project_points_3d_to_2d


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
