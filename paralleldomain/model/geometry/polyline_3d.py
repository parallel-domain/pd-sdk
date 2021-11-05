from dataclasses import dataclass
from typing import List

import numpy as np

from paralleldomain.model.geometry.point_3d import Point3DGeometry


@dataclass
class Line3DGeometry:
    """Represents a 3D Line.

    Args:
        start: :attr:`~.Line3D.start`
        end: :attr:`~.Line3D.end`

    Attributes:
        start: the 3D start point of the line in image coordinates
        end: the 3D end point of the line in image coordinates
    """

    start: Point3DGeometry
    end: Point3DGeometry

    def numpy(self):
        """Returns the start and end coordinates as a numpy array with shape (2 x 3)."""
        return np.vstack([self.start.numpy(), self.end.numpy()])


@dataclass
class Polyline3DGeometry:
    """A polyline made of a collection of 3D Lines

    Args:
        lines: :attr:`~.Polyline3D.lines`

    Attributes:
        lines: Ordered list of :obj:`Line3D` instances
    """

    lines: List[Line3DGeometry]

    def numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 3)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 3))
        elif num_lines == 1:
            return self.lines[0].numpy()
        else:
            return np.vstack([ll.numpy()[0] for ll in self.lines[:-1]] + [self.lines[-1].numpy()[1]])
