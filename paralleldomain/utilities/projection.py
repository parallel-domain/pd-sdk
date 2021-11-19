import math
from typing import Optional

import cv2
import numpy as np

from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_PD_FISHEYE


class DistortionLookupTable(np.ndarray):
    """Container object for distortion lookup tables used in distortion model `pd_fisheye`.
    Can be accessed like any `np.ndarray`"""

    @classmethod
    def from_ndarray(cls, data: np.ndarray) -> "DistortionLookupTable":
        """Takes a `np.ndarray` and creates a :obj:`DistortionLookupTable` instance from it.

        Args:
            data: `np.ndarray` of shape (N x 2). Array will be sorted by first column. Last two rows will be used to
                extrapolate for maximum valid angle value of Pi.

        Returns:
            Distortion lookup table to be used with projection functions using distortion model `pd_fisheye`.
        """
        data_sorted = data[np.argsort(data[:, 0])].astype(np.float)
        if data_sorted[:, 0].max() < math.pi:
            extrapolated_alpha = (math.pi - data_sorted[-2, 0]) / (data_sorted[-1, 0] - data_sorted[-2, 0])
            extrapolated_theta_d = data_sorted[-2, 1] + extrapolated_alpha * (data_sorted[-1, 1] - data_sorted[-2, 1])
            data_sorted = np.vstack([data_sorted, np.array([math.pi, extrapolated_theta_d])])
        return data_sorted.view(cls)


def _project_points_3d_to_2d_pd_fisheye(
    k_matrix: np.ndarray,
    points_3d: np.ndarray,
    distortion_lookup: DistortionLookupTable,
) -> np.ndarray:
    xy_prime = points_3d[:, [0, 1]]
    r = np.linalg.norm(xy_prime, axis=1)

    theta = np.arctan2(r, points_3d[:, 2])
    theta_d = np.interp(x=theta, xp=distortion_lookup[:, 0], fp=distortion_lookup[:, 1])

    r_d = (theta_d / r).reshape(-1, 1)

    xy_double_prime = r_d * xy_prime
    xy_double_prime[np.isnan(xy_double_prime)] = 0.0
    xy_double_prime_one = np.ones(shape=(len(xy_double_prime), 1))

    uv = (k_matrix @ np.hstack([xy_double_prime, xy_double_prime_one]).T).T[:, :2]

    return uv


def project_points_3d_to_2d(
    k_matrix: np.ndarray,
    camera_model: str,
    points_3d: np.ndarray,
    distortion_parameters: Optional[np.ndarray] = None,
    distortion_lookup: Optional[DistortionLookupTable] = None,
) -> np.ndarray:
    """Projects an array of 3D points in Cartesian coordinates onto an image plane.

    Args:
        k_matrix: Camera intrinsic matrix. Definition can be found in
            `OpenCV documentation <https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html>`_.
        camera_model: One of `opencv_pinhole` or `opencv_fisheye`.
            More details in :obj:`~.model.sensor.CameraModel`.
        points_3d: A matrix with dimensions (nx3) containing the points.
            Points must be already in the camera's coordinate system.
        distortion_parameters: Array of applicable distortion parameters for
            distortion models `opencv_pinhole` and `opencv_fisheye`.
        distortion_lookup: Table of undistorted and distorted angles. Required for `pd_fisheye` model.

    Returns:
        A matrix with dimensions (nx2) containing the point projections. :obj:`dtype` is :obj:`float` and values need
        to be rounded to integers by the user to receive actual pixel coordinates. Includes all points, independent
        if they are on the image plane or outside. Points behind the image plane will be projected, too. If values are
        not valid (e.g., for undistorted pinhole cameras), filtering needs to be applied by the calling function.
    """

    k_matrix = k_matrix.reshape(3, 3).astype(np.float)
    points_3d = points_3d.reshape(-1, 3).astype(np.float)

    if distortion_parameters is not None:
        distortion_parameters = distortion_parameters.reshape(1, -1).astype(np.float)

    if camera_model == CAMERA_MODEL_OPENCV_PINHOLE:
        uv, _ = cv2.projectPoints(
            objectPoints=points_3d,
            rvec=(0, 0, 0),  # already in camera sensor coordinate system
            tvec=(0, 0, 0),  # already in camera sensor coordinate system
            cameraMatrix=k_matrix,
            distCoeffs=distortion_parameters,
        )
    elif camera_model == CAMERA_MODEL_OPENCV_FISHEYE:
        points_3d = np.expand_dims(points_3d, -2)  # cv2.fisheye.projectPoints expects dimensions (N x 1 x 3)
        uv, _ = cv2.fisheye.projectPoints(
            objectPoints=points_3d,
            rvec=(0, 0, 0),  # already in camera sensor coordinate system
            tvec=(0, 0, 0),  # already in camera sensor coordinate system
            K=k_matrix,
            D=distortion_parameters,
        )
    elif camera_model == CAMERA_MODEL_PD_FISHEYE:
        uv = _project_points_3d_to_2d_pd_fisheye(
            k_matrix=k_matrix,
            points_3d=points_3d,
            distortion_lookup=distortion_lookup,
        )
    else:
        raise NotImplementedError(f'Distortion Model "{camera_model}" not implemented.')

    return uv.reshape(-1, 2)
