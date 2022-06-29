import abc
import math
from typing import Optional

import cv2
import numpy as np

from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_PD_FISHEYE
from paralleldomain.utilities.mask import lookup_values


class DistortionLookup:
    @abc.abstractmethod
    def evaluate_at(self, theta: np.ndarray) -> np.ndarray:
        pass


class DistortionLookupTable(DistortionLookup, np.ndarray):
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

    def evaluate_at(self, theta: np.ndarray) -> np.ndarray:
        return np.interp(x=theta, xp=self[:, 0], fp=self[:, 1])


def _project_points_3d_to_2d_pd_fisheye(
    k_matrix: np.ndarray,
    points_3d: np.ndarray,
    distortion_lookup: DistortionLookup,
) -> np.ndarray:
    xy_prime = points_3d[:, [0, 1]]
    r = np.linalg.norm(xy_prime, axis=1)

    theta = np.arctan2(r, points_3d[:, 2])
    theta_d = distortion_lookup.evaluate_at(theta=theta)

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
    distortion_lookup: Optional[DistortionLookup] = None,
) -> np.ndarray:
    """Projects an array of 3D points in Cartesian coordinates onto an image plane.

    Args:
        k_matrix: Camera intrinsic matrix. Definition can be found in
            `OpenCV documentation <https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html>`_.
        camera_model: One of `opencv_pinhole`, `opencv_fisheye`, `pd_fisheye`.
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


def project_points_2d_to_3d(
    k_matrix: np.ndarray,
    camera_model: str,
    points_2d: np.ndarray,
    depth: np.ndarray,
    distortion_parameters: Optional[np.ndarray] = None,
    distortion_lookup: Optional[DistortionLookup] = None,
    interpolate: bool = True,
) -> np.ndarray:
    """Maps image plane coordinates to 3D points in Cartesian coordinates.

    Args:
        k_matrix: Camera intrinsic matrix. Definition can be found in
            `OpenCV documentation <https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html>`_.
        camera_model: One of `opencv_pinhole`, `opencv_fisheye`, `pd_fisheye`.
            More details in :obj:`~.model.sensor.CameraModel`.
        points_2d: A matrix with dimensions (nx2) containing the points.
            Points must be in image coordinate system (x,y).
        depth: Depth mask with the same dimensions as the image canvas.
        distortion_parameters: Array of applicable distortion parameters for
            distortion models `opencv_pinhole` and `opencv_fisheye`.
        distortion_lookup: Table of undistorted and distorted angles. Required for `pd_fisheye` model.
        interpolate: When points are not exactly on an image pixel, apply bi-linear interpolation to estimate
            the corresponding depth value. Default: True.
    Returns:
        A matrix with dimensions (nx3) containing the point projections in 3D using the provided depth mask.
    """

    k_matrix = k_matrix.reshape(3, 3).astype(np.float)
    points_2d = points_2d.reshape(-1, 2).astype(np.float)

    # Uncomment when OpenCV with distortion reprojection is being implemented
    # if distortion_parameters is not None:
    #     distortion_parameters = distortion_parameters.reshape(1, -1).astype(np.float)

    depth_for_points_2d = lookup_values(mask=depth, x=points_2d[:, 0], y=points_2d[:, 1], interpolate=interpolate)
    if camera_model == CAMERA_MODEL_OPENCV_PINHOLE:
        points_3d = (
            np.linalg.inv(k_matrix) @ np.hstack([points_2d, np.ones(shape=(len(points_2d), 1))]).T
        ).T * depth_for_points_2d

    elif camera_model == CAMERA_MODEL_PD_FISHEYE:
        points_3d_distorted = (np.linalg.inv(k_matrix) @ np.hstack([points_2d, np.ones(shape=(len(points_2d), 1))]).T).T

        xy_prime = points_3d_distorted[:, [0, 1]]
        theta_d = np.linalg.norm(xy_prime, axis=1)
        theta = np.interp(x=theta_d, xp=distortion_lookup[:, 1], fp=distortion_lookup[:, 0])

        r = np.tan(theta)

        xy_double_prime = (r / theta_d).reshape(-1, 1) * xy_prime
        xy_double_prime[np.isnan(xy_double_prime)] = 0.0
        xy_double_prime_one = np.ones(shape=(len(xy_double_prime), 1))

        points_3d = np.hstack([xy_double_prime, xy_double_prime_one]) * depth_for_points_2d
    else:
        raise NotImplementedError(f'Distortion Model "{camera_model}" not implemented.')

    return points_3d.reshape(-1, 3)


def points_2d_inside_image(
    width: int,
    height: int,
    camera_model: str,
    points_2d: np.ndarray,
    points_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns the indices for an array of 2D image points that are inside the image canvas.

    Args:
        width: Pixel width of the image canvas.
        height: Pixel height of the image canvas.
        camera_model: One of `opencv_pinhole`, `opencv_fisheye`, `pd_fisheye`.
            More details in :obj:`~.model.sensor.CameraModel`.
        points_2d: A matrix with dimensions (nx2) containing the points that should be tested
             if inside the image canvas. Points must be in image coordinate system (x,y).
        points_3d: Optional array of size (nx3) which provides the 3D camera coordinates for each point. Required for
            camera models `opencv_pinhole` and `opencv_fisheye`.
    Returns:
        An array with dimensions (n,).
    """

    if camera_model in (CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_OPENCV_FISHEYE) and points_3d is None:
        raise ValueError(f"`points_3d` must be provided for camera model {camera_model}")
    if len(points_2d) != len(points_3d):
        raise ValueError(
            f"Mismatch in length between `points_2d` and `points_3d` with {len(points_2d)} vs. {len(points_3d)}"
        )

    return np.where(
        (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < width)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < height)
        & (points_3d[:, 2] > 0 if camera_model in (CAMERA_MODEL_OPENCV_PINHOLE, CAMERA_MODEL_OPENCV_FISHEYE) else True)
    )
