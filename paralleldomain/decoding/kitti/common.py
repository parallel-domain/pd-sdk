from functools import lru_cache

import numpy as np
from typing import Optional, List, Tuple, Dict, Union

from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation


# def rot_2d(points_2d: np.ndarray, theta: float):
#     c, s = np.cos(theta), np.sin(theta)
#     rot_mat = np.array(((c, -s), (s, c)))
#     rot_points = (rot_mat @ points_2d.T).T
#     return rot_points


@lru_cache(maxsize=10)
def _cached_point_cloud(pointcloud_file: str) -> np.ndarray:
    with AnyPath(pointcloud_file).open(mode="rb") as fp:
        point_cloud_data = np.frombuffer(fp.read(), dtype="<f4")  # little-endian float32

    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # 4 for KITTI
    return point_cloud_data


@lru_cache(maxsize=10)
def _cached_sensor_frame_calibrations(scene_folder_path: AnyPath, frame_id: str) -> Dict[str, np.ndarray]:
    """

    Calibrations available:
    P0: todo
    P1: todo
    P2: todo
    P3: todo
    R0_rect: todo
    Tr_velo_to_cam: velodyne (LiDAR - FLU) to camera (RDF)
    Tr_imu_to_velo: todo

    """
    calibration_file = scene_folder_path / "calib" / f"{frame_id}.txt"
    calibration_transforms = {}
    with calibration_file.open(mode="r") as fc:
        for line in fc.readlines():
            split_line = line.rstrip().split(" ")
            calib_name = split_line[0].split(":")[0]

            if calib_name != "R0_rect" and calib_name != "":
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :] = np.reshape(split_line[1:], (3, 4)).astype(float)
            elif calib_name != "":
                transformation_matrix = np.reshape(split_line[1:], (3, 3)).astype(float)
            else:
                break
            calibration_transforms[calib_name] = transformation_matrix

    return calibration_transforms
