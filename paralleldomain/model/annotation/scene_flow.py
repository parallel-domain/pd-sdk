from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class SceneFlow(Annotation):
    """Represents an Scene Flow mask for a point cloud.

    Args:
        vectors: :attr:`paralleldomain.model.annotation.scene_flow.SceneFlow.vectors`

    Attributes:
        vectors: Matrix of shape `(N x 3)`, , where `N` is the number of points of the corresponding
            point cloud. The second axis contains the x, y and z offset to the position the sampled point will be at
            in the next frame. Note: This exact position might not be sampled by a Lidar in the next frame!

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

    vectors: np.ndarray

    def __sizeof__(self):
        return self.vectors.nbytes
