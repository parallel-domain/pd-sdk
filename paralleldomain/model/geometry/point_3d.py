from dataclasses import dataclass

import numpy as np

from paralleldomain.utilities.transformation import Transformation


@dataclass
class Point3DGeometry:
    """Represents a 3D Point.

    Args:
        x: :attr:`~.Point3D.x`
        y: :attr:`~.Point3D.y`
        z: :attr:`~.Point3D.z`
        class_id: :attr:`~.Point3D.class_id`
        instance_id: :attr:`~.Point3D.instance_id`
        attributes: :attr:`~.Point3D.attributes`

    Attributes:
        x: coordinate along x-axis in sensor coordinates
        y: coordinate along y-axis in sensor coordinates
        z: coordinate along z-axis in sensor coordinates
        class_id: Class ID of the point. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    x: float
    y: float
    z: float

    def to_numpy(self):
        """Returns the coordinates as a numpy array with shape (1 x 3)."""
        return np.array([[self.x, self.y, self.z]])

    def transform(self, tf: Transformation) -> "Point3DGeometry":
        tf_point = (tf @ np.array([self.x, self.y, self.z, 1]))[:3]
        return Point3DGeometry(x=tf_point[0], y=tf_point[1], z=tf_point[2])

    @classmethod
    def from_numpy(cls, point: np.ndarray):
        pt = point.reshape(-3)
        return cls(x=pt[0], y=pt[1], z=pt[2])
