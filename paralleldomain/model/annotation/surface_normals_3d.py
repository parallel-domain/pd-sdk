from dataclasses import dataclass

import numpy as np

from paralleldomain.model.annotation.common import Annotation


@dataclass
class SurfaceNormals3D(Annotation):
    """Represents a mask of surface normals for a point cloud.

    Args:
        normals: :attr:`~.SurfaceNormals3D.vectors`

    Attributes:
        normals: Matrix of shape `(N x 3)`, , where `N` is the number of points of the corresponding
            point cloud. The second axis contains the x, y and z normal direction of the surface the corresponding point
            was sampled from.

    Example:
        Using the Surface Normal mask in combination with :attr:`.PointCloud.xyz` to visualize the
        normals of each point.
        ::

            lidar_frame: LidarSensorFrame = ...  # get any lidars's SensorFrame

            sur_normals = lidar_frame.get_annotations(AnnotationTypes.SurfaceNormals3D)
            xyz = lidar_frame.point_cloud.xyz
            normal_points = xyz + sur_normals.normals
            normal_line_points = np.concatenate([xyz, normal_points], axis=-1).reshape((2 * len(xyz), 3))
            line_connections = np.arange(2 * len(xyz)).reshape((-1, 2))

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            colors = [[1, 0, 0] for i in range(len(line_connections))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(normal_line_points),
                lines=o3d.utility.Vector2iVector(line_connections),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([pcd, line_set])
    """

    normals: np.ndarray

    def __sizeof__(self):
        return self.normals.nbytes
