from dataclasses import dataclass
from typing import List

import numpy as np

from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.utilities.geometry import interpolate_points
from paralleldomain.utilities.transformation import Transformation


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

    def to_numpy(self):
        """Returns the start and end coordinates as a numpy array with shape (2 x 3)."""
        return np.vstack([self.start.to_numpy(), self.end.to_numpy()])

    def transform(self, tf: Transformation) -> "Line3DGeometry":
        return Line3DGeometry(start=self.start.transform(tf=tf), end=self.end.transform(tf=tf))

    @classmethod
    def from_numpy(cls, points: np.ndarray) -> "Line3DGeometry":
        points = points.reshape(2, 3)
        return Line3DGeometry(start=Point3DGeometry.from_numpy(points[0]), end=Point3DGeometry.from_numpy(points[1]))


@dataclass
class Polyline3DGeometry:
    """A polyline made of a collection of 3D Lines

    Args:
        lines: :attr:`~.Polyline3D.lines`

    Attributes:
        lines: Ordered list of :obj:`Line3D` instances
    """

    lines: List[Line3DGeometry]

    def to_numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 3)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 3))
        elif num_lines == 1:
            return self.lines[0].to_numpy()
        else:
            return np.vstack([ll.to_numpy()[0] for ll in self.lines] + [self.lines[-1].to_numpy()[1]])

    def transform(self, tf: Transformation) -> "Polyline3DGeometry":
        return Polyline3DGeometry(lines=[ll.transform(tf=tf) for ll in self.lines])

    @classmethod
    def from_numpy(cls, points: np.ndarray, **kwargs) -> "Polyline3DGeometry":
        points = points.reshape(-1, 3)
        point_pairs = np.hstack([points[:-1], points[1:]])
        kwargs["lines"] = np.apply_along_axis(Line3DGeometry.from_numpy, point_pairs)
        return cls(**kwargs)
