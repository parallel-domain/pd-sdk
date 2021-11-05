from dataclasses import dataclass
from typing import List

import numpy as np

from paralleldomain.model.geometry.point_2d import Point2DGeometry


@dataclass
class Line2DGeometry:
    """Represents a 2D Line.

    Args:
        start: :attr:`~.Line2D.start`
        end: :attr:`~.Line2D.end`
        class_id: :attr:`~.Line2D.class_id`
        instance_id: :attr:`~.Line2D.instance_id`
        attributes: :attr:`~.Line2D.attributes`

    Attributes:
        start: the 2D start point of the line in image coordinates
        end: the 2D end point of the line in image coordinates
        directed: whether the line is directed from start to end (if False the line is bi-directional)
        class_id: Class ID of the line. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    start: Point2DGeometry
    end: Point2DGeometry

    def numpy(self):
        """Returns the start and end coordinates as a numpy array with shape (2 x 2)."""
        return np.vstack([self.start.numpy(), self.end.numpy()])


@dataclass
class Polyline2DGeometry:
    """A polyline made of a collection of 2D Lines

    Args:
        lines: :attr:`~.Polyline2D.lines`

    Attributes:
        lines: Ordered list of :obj:`Line2D` instances
    """

    lines: List[Line2DGeometry]

    def numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 2)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 2))
        elif num_lines == 1:
            return self.lines[0].numpy()
        else:
            return np.vstack([ll.numpy()[0] for ll in self.lines[:-1]] + [self.lines[-1].numpy()[1]])
