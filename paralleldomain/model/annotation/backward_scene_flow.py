from dataclasses import dataclass
from typing import Optional

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class BackwardSceneFlow(Annotation):
    """
    Backwards Scene Flow mask for a point cloud.

    Args:
        vectors: :attr:`BackwardSceneFlow.vectors`
        valid_mask: :attr:`BackwardSceneFlow.valid_mask`

    Attributes:
        vectors: Array containing the x, y, z components of the vector denoting the movement of each point in the
            point cloud between the frame and the corresponding pixel at the previous timestep. Vector is calculated by
            subtracting point position at time i from point position from time i-1
        valid_mask: Array containing a boolean value which is True when that point contains a valid backwards scene
            flow vector at that points array location in :attr::`BackwardSceneFlow.vectors`, and False when there is no
            valid vector

    Example:
        Using the Scene Flow vector mask in combination with :attr:`AnnotationTypes.PointCloud.xyz` to get a view of the
        previous frame ::

            >>> lidar_frame: LidarSensorFrame = ...  # get any lidars's SensorFrame
            >>>
            >>> flow = lidar_frame.get_annotations(AnnotationTypes.BackwardSceneFlow)
            >>> xyz = lidar_frame.point_cloud.xyz
            >>> previous_frame_xyz = xyz + flow.vectors
            >>>
            >>> import open3d as o3d
            >>> pcd = o3d.geometry.PointCloud()
            >>> pcd.points = o3d.utility.Vector3dVector(previous_frame_xyz)
            >>> o3d.visualization.draw_geometries([pcd])
    """

    vectors: np.ndarray
    valid_mask: Optional[np.ndarray] = None

    def __sizeof__(self):
        total_size = 0
        for vec in [self.vectors, self.valid_mask]:
            if vec is not None:
                total_size += vec.nbytes
        return total_size
