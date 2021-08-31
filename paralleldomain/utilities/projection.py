import cv2
import numpy as np

from paralleldomain.model.sensor import CameraModel, SensorFrame


def project_points_3d_to_2d(camera_frame: SensorFrame, points_3d: np.ndarray) -> np.ndarray:
    """Projects an array of 3D points in Cartesian coordinates onto an image plane.

    Args:
        camera_frame: The camera frame the points will be projected onto.
        points_3d: A matrix with dimensions (nx3) containing the points.
            Points must be already in the selected camera frame's coordinate system.

    Returns:
        A matrix with dimensions (nx2) containing the point projections. :obj:`dtype` is :obj:`float` and values need
        to be rounded to integers by the user to receive actual pixel coordinates. Includes all points, independent
        if they are on the image plane or outside.
    """

    intrinsic = camera_frame.intrinsic
    K = np.array(
        [
            [intrinsic.fx, intrinsic.skew, intrinsic.cx],
            [0, intrinsic.fy, intrinsic.cy],
            [0, 0, 1],
        ]
    )

    r_vec = t_vec = np.array([0, 0, 0]).astype(float)

    if intrinsic.camera_model == CameraModel.OPENCV_PINHOLE:
        D = np.array(
            [
                intrinsic.k1,
                intrinsic.k2,
                intrinsic.p1,
                intrinsic.p2,
                intrinsic.k3,
                intrinsic.k4,
                intrinsic.k5,
                intrinsic.k6,
            ]
        )
        uv, _ = cv2.projectPoints(points_3d, r_vec, t_vec, K, D)
    elif intrinsic.camera_model == CameraModel.OPENCV_FISHEYE:
        D = np.array(
            [
                intrinsic.k1,
                intrinsic.k2,
                intrinsic.k3,
                intrinsic.k4,
            ]
        )
        uv, _ = cv2.fisheye.projectPoints(points_3d, r_vec, t_vec, K, D)
    else:
        raise ValueError(f"Unsupported camera model {intrinsic.camera_model}")

    return uv.reshape(-1, 2)
