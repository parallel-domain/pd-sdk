from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation

# def rot_2d(points_2d: np.ndarray, theta: float):
#     c, s = np.cos(theta), np.sin(theta)
#     rot_mat = np.array(((c, -s), (s, c)))
#     rot_points = (rot_mat @ points_2d.T).T
#     return rot_points


@lru_cache(maxsize=10)
def _cached_point_cloud(pointcloud_file: str, point_cloud_dim: int) -> np.ndarray:
    with AnyPath(pointcloud_file).open(mode="rb") as fp:
        point_cloud_data = np.frombuffer(fp.read(), dtype="<f4")  # little-endian float32

    # Default Kitti is 4 dims (x, y, z, intensity)
    point_cloud_data = np.reshape(point_cloud_data, (-1, point_cloud_dim))
    return point_cloud_data


@lru_cache(maxsize=10)
def _cached_sensor_frame_calibrations(scene_folder_path: AnyPath, frame_id: str) -> Dict[str, np.ndarray]:
    """
    Kitti data labels are assumed to be in (RDF) Camera coordinate systems. PD-SDK assumes labels are in (FLU) sensor
    coordinate systems. Pointcloud/velodyne is in (FLU) Sensor coordinate system.

    Calibrations available:
    P0: todo
    P1: todo
    P2: todo
    P3: todo
    R0_rect: todo
    Tr_velo_to_cam: velodyne (LiDAR - FLU) to camera (RDF)
    Tr_imu_to_velo: todo

    If no calibration is used - there is not /calib folder, an identity transformation matrix is used.
      - Tr_velo_to_cam: transforms data from RDF coordinates to FLU coordinate
    """
    calibration_file = scene_folder_path / "calib" / f"{frame_id}.txt"
    calibration_transforms = {}
    if calibration_file.exists():
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
                    # we reach end of file
                    break
                calibration_transforms[calib_name] = transformation_matrix
    else:
        calibration_transforms = {
            "P0": None,
            "P1": None,
            "P2": None,
            "P3": None,
            "R0_rect": None,
            "Tr_velo_to_cam": np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),  # RDF to FLU
            "Tr_imu_to_velo": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # FLU
        }

    return calibration_transforms
