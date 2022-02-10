from dataclasses import dataclass
from typing import Generic, List, TypeVar

import numpy as np

from paralleldomain.model.geometry.point_3d import Point3DBaseGeometry, Point3DGeometry
from paralleldomain.utilities.geometry import interpolate_points
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T", int, float)


@dataclass
class Line3DBaseGeometry(Generic[T]):
    """Represents a 3D Line.

    Args:
        start: :attr:`~.Line3DBaseGeometry.start`
        end: :attr:`~.Line3DBaseGeometry.end`

    Attributes:
        start: the 3D start point of the line in image coordinates
        end: the 3D end point of the line in image coordinates
    """

    start: Point3DBaseGeometry[T]
    end: Point3DBaseGeometry[T]

    @property
    def direction(self) -> Point3DBaseGeometry[T]:
        """Returns the directional vector of the line."""
        return self.end - self.start

    @property
    def length(self) -> float:
        """Returns the length of the line."""
        return np.linalg.norm(self.direction.to_numpy().reshape(3))

    def to_numpy(self):
        """Returns the start and end coordinates as a numpy array with shape (2 x 3)."""
        return np.vstack([self.start.to_numpy(), self.end.to_numpy()])

    def transform(self, tf: Transformation) -> "Line3DBaseGeometry[T]":
        return Line3DBaseGeometry[T](start=self.start.transform(tf=tf), end=self.end.transform(tf=tf))

    @classmethod
    def from_numpy(cls, points: np.ndarray) -> "Line3DBaseGeometry[T]":
        points = points.reshape(2, 3)
        return Line3DBaseGeometry[T](
            start=Point3DBaseGeometry[T].from_numpy(points[0]), end=Point3DBaseGeometry[T].from_numpy(points[1])
        )


class Line3DGeometry(Line3DBaseGeometry[float]):
    pass


@dataclass
class Polyline3DBaseGeometry(Generic[T]):
    """A polyline made of a collection of 3D Lines

    Args:
        lines: :attr:`~.Polyline3DBaseGeometry.lines`

    Attributes:
        lines: Ordered list of :obj:`Line3DBaseGeometry` instances
    """

    lines: List[Line3DBaseGeometry[T]]

    @property
    def length(self):
        """Returns the length of the line."""
        return sum([ll.length for ll in self.lines])

    def to_numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 3)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 3))
        elif num_lines == 1:
            return self.lines[0].to_numpy()
        else:
            return np.vstack([ll.to_numpy()[0] for ll in self.lines] + [self.lines[-1].to_numpy()[1]])

    def transform(self, tf: Transformation) -> "Polyline3DBaseGeometry[T]":
        return Polyline3DBaseGeometry[T](lines=[ll.transform(tf=tf) for ll in self.lines])

    @classmethod
    def from_numpy(cls, points: np.ndarray, **kwargs) -> "Polyline3DBaseGeometry[T]":
        points = points.reshape(-1, 3)
        point_pairs = np.hstack([points[:-1], points[1:]])
        kwargs["lines"] = np.apply_along_axis(Line3DGeometry.from_numpy, point_pairs)
        return cls(**kwargs)


class Polyline3DGeometry(Polyline3DBaseGeometry[float]):
    pass
