import cv2
import numpy as np

from paralleldomain.model.sensor import SensorFrame


def project_points_3d_to_2d(camera_frame: SensorFrame, points_3d: np.ndarray) -> np.ndarray:
    intrinsic = camera_frame.intrinsic
    K = np.array(
        [
            [intrinsic.fx, intrinsic.skew, intrinsic.cx],
            [0, intrinsic.fy, intrinsic.cy],
            [0, 0, 1],
        ]
    )

    r_vec = t_vec = np.array([0, 0, 0]).astype(float)

    if intrinsic.camera_model == "brown_conrady":
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
    elif intrinsic.camera_model == "fisheye":
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
