import cv2
import numpy as np


def project_points_3d_to_2d(
    k_matrix: np.ndarray, distortion_parameters: np.ndarray, distortion_model: str, points_3d: np.ndarray
) -> np.ndarray:
    """Projects an array of 3D points in Cartesian coordinates onto an image plane.

    Args:
        k_matrix: Camera intrinsic matrix. Definition can be found in
            `OpenCV documentation <https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html>`_.
        distortion_parameters: Array of applicable distortion parameters for distortion model.
        distortion_model: One of `opencv_pinhole` or `opencv_fisheye`.
            More details in :obj:`~.model.sensor.CameraModel`.
        points_3d: A matrix with dimensions (nx3) containing the points.
            Points must be already in the camera's coordinate system.

    Returns:
        A matrix with dimensions (nx2) containing the point projections. :obj:`dtype` is :obj:`float` and values need
        to be rounded to integers by the user to receive actual pixel coordinates. Includes all points, independent
        if they are on the image plane or outside.
    """

    r_vec = t_vec = np.array([0, 0, 0]).astype(float)
    k_matrix = k_matrix.reshape(3, 3)
    distortion_parameters = distortion_parameters.reshape(-1)

    if distortion_model == "opencv_pinhole":
        uv, _ = cv2.projectPoints(points_3d, r_vec, t_vec, k_matrix, distortion_parameters)
    elif distortion_model == "opencv_fisheye":
        uv, _ = cv2.fisheye.projectPoints(points_3d, r_vec, t_vec, k_matrix, distortion_parameters)
    else:
        raise NotImplementedError(f'Distortion Model "{distortion_model}" not implemented.')

    return uv.reshape(-1, 2)
