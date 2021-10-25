from typing import Optional

import cv2
import numpy as np

from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_PD_FISHEYE


class DistortionLookupTable(np.ndarray):
    @classmethod
    def from_ndarray(cls, data: np.ndarray) -> "DistortionLookupTable":
        data_sorted = data[np.argsort(data[:, 0])].astype(np.float32)
        return data_sorted.view(cls)


def project_points_3d_to_2d_pd_fisheye(
    k_matrix: np.ndarray,
    points_3d: np.ndarray,
    distortion_lookup: DistortionLookupTable,
) -> np.ndarray:
    xy_prime = points_3d[:, [0, 1]] / points_3d[:, [2]]  # x/z y/z

    r = np.linalg.norm(xy_prime, axis=1)  # sqrt(x^2 + y^2)
    theta = np.arctan(r)  # is that right?
    theta_d = np.interp(
        x=theta, xp=distortion_lookup[:, 0], fp=distortion_lookup[:, 1]
    )  # column 0: theta, column 1: theta_d
    r_d = theta_d.reshape(-1, 1)  # theta_d = r_d ?

    xy_double_prime = xy_prime * r_d  # multiply with r_d

    uv = xy_double_prime * k_matrix[0, [0, 2]] + k_matrix[1, [1, 2]]  # multiply with focal length and add offset

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
        distortion_parameters: Array of applicable distortion parameters for distortion model.
        camera_model: One of `opencv_pinhole` or `opencv_fisheye`.
            More details in :obj:`~.model.sensor.CameraModel`.
        points_3d: A matrix with dimensions (nx3) containing the points.
            Points must be already in the camera's coordinate system.

    Returns:
        A matrix with dimensions (nx2) containing the point projections. :obj:`dtype` is :obj:`float` and values need
        to be rounded to integers by the user to receive actual pixel coordinates. Includes all points, independent
        if they are on the image plane or outside.
    """

    r_vec = t_vec = np.array([0, 0, 0]).astype(np.float32)
    k_matrix = k_matrix.reshape(3, 3).astype(np.float32)
    points_3d = points_3d.reshape(-1, 3).astype(np.float32)

    if distortion_parameters is not None:
        distortion_parameters = distortion_parameters.reshape(1, -1).astype(np.float32)

    if camera_model == CAMERA_MODEL_OPENCV_PINHOLE:
        uv, _ = cv2.projectPoints(points_3d, r_vec, t_vec, k_matrix, distortion_parameters)
    elif camera_model == CAMERA_MODEL_OPENCV_FISHEYE:
        points_3d = np.expand_dims(points_3d, -2)  # cv2.fisheye.projectPoints expects dimensions (N x 1 x 3)
        uv, _ = cv2.fisheye.projectPoints(points_3d, r_vec, t_vec, k_matrix, distortion_parameters)
    elif camera_model == CAMERA_MODEL_PD_FISHEYE:
        uv = project_points_3d_to_2d_pd_fisheye(
            k_matrix=k_matrix, points_3d=points_3d, distortion_lookup=distortion_lookup
        )
    else:
        raise NotImplementedError(f'Distortion Model "{camera_model}" not implemented.')

    return uv.reshape(-1, 2)
