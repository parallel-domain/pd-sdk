from dataclasses import dataclass

import numpy as np


@dataclass
class Point3D:
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

    def numpy(self):
        """Returns the coordinates as a numpy array with shape (1 x 3)."""
        return np.array([[self.x, self.y, self.z]])
