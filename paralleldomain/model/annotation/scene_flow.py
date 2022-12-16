from dataclasses import dataclass
from typing import Optional

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class SceneFlow(Annotation):
    """Represents a Scene Flow mask for a point cloud.

    Args:
        vectors: :attr:`paralleldomain.model.annotation.scene_flow.SceneFlow.vectors`
        backward_vectors: :attr:`paralleldomain.model.annotation.scene_flow.SceneFlow.backward_vectors`

    Attributes:
        vectors: Matrix of shape `(N x 3)`, , where `N` is the number of points of the corresponding
            point cloud. The second axis contains the x, y and z offset to the position the sampled point will be at
            in the next frame. Note: This exact position might not be sampled by a Lidar in the next frame!
        backward_vectors: Matrix of shape `(N x 3)`, , where `N` is the number of points of the corresponding
            point cloud. The second axis contains the x, y and z offset to the position the sampled point will be at
            in the previous frame. Note: This exact position might not be sampled by a Lidar in the next frame!

    Example:
        Using the Scene Flow vector mask in combination with :attr:`.PointCloud.xyz` to get a view of the next frame.
        ::

            lidar_frame: LidarSensorFrame = ...  # get any lidars's SensorFrame

            flow = lidar_frame.get_annotations(AnnotationTypes.SceneFlow)
            xyz = lidar_frame.point_cloud.xyz
            next_frame_xyz = xyz + flow.vectors

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(next_frame_xyz)
            o3d.visualization.draw_geometries([pcd])
    """

    vectors: Optional[np.ndarray] = None
    backward_vectors: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.vectors is None and self.backward_vectors is None:
            raise ValueError("Invalid SceneFlow annotation! Either vectors or backward_vectors has to contain a value!")

    def __sizeof__(self):
        return self.vectors.nbytes
