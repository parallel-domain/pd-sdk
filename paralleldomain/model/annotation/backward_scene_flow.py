from dataclasses import dataclass
from typing import Optional

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class BackwardSceneFlow(Annotation):
    """Represents a Backwards Scene Flow mask for a point cloud.

    Args:
        vectors: :attr:`paralleldomain.model.annotation.scene_flow.BackwardSceneFlow.vectors`
        valid_mask: :attr:`paralleldomain.model.annotation.scene_flow.BackwardSceneFlow.valid_mask`

    Attributes:
        vectors: Matrix of shape `(N x 3)`, where `N` is the number of points of the corresponding
            point cloud. The second axis contains the x, y and z offset to the position the sampled point will be at
            in the previous frame. Note: This exact position might not be sampled by a Lidar in the previous frame!
        valid_mask: Matrix of shape `(N X 1)` with values {0,1}. 1 indicates a point with a valid flow label in the
            vectors attribute. 0 indicates no groundtruth flow at that pixel.

    Example:
        Using the Scene Flow vector mask in combination with :attr:`.PointCloud.xyz` to get a view of the
        previous frame.
        ::

            lidar_frame: LidarSensorFrame = ...  # get any lidars's SensorFrame

            flow = lidar_frame.get_annotations(AnnotationTypes.BackwardSceneFlow)
            xyz = lidar_frame.point_cloud.xyz
            previous_frame_xyz = xyz + flow.vectors

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(previous_frame_xyz)
            o3d.visualization.draw_geometries([pcd])
    """

    vectors: np.ndarray
    valid_mask: Optional[np.ndarray] = None

    def __sizeof__(self):
        total_size = 0
        for vec in [self.vectors, self.valid_mask]:
            if vec is not None:
                total_size += vec.nbytes
        return total_size
