from dataclasses import dataclass

import numpy as np


@dataclass
class Point2DGeometry:
    """Represents a 2D Point.

    Args:
        x: :attr:`~.Point2D.x`
        y: :attr:`~.Point2D.y`
        class_id: :attr:`~.Point2D.class_id`
        instance_id: :attr:`~.Point2D.instance_id`
        attributes: :attr:`~.Point2D.attributes`

    Attributes:
        x: coordinate along x-axis in image pixels
        y: coordinate along y-axis in image pixels
        class_id: Class ID of the point. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    x: int
    y: int

    def numpy(self):
        """Returns the coordinates as a numpy array with shape (1 x 2)."""
        return np.array([[self.x, self.y]])
