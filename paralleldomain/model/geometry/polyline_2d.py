from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import numpy as np

from paralleldomain.model.geometry.point_2d import Point2DBaseGeometry

T = TypeVar("T", int, float)


class CollinearLinesException(Exception):
    pass


@dataclass
class Line2DBaseGeometry(Generic[T]):
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

    start: Point2DBaseGeometry[T]
    end: Point2DBaseGeometry[T]

    @property
    def direction(self) -> Point2DBaseGeometry[T]:
        """Returns the directional vector of the line."""
        return self.end - self.start

    @property
    def length(self) -> float:
        """Returns the length of the line."""
        return np.linalg.norm(self.direction.to_numpy().reshape(2))

    @property
    def slope(self) -> float:
        """Returns the slope of the line. Returns `np.inf` for vertical lines."""
        try:
            return self.direction.y / self.direction.x
        except ZeroDivisionError:
            return np.inf

    def to_numpy(self) -> np.ndarray:
        """Returns the start and end coordinates as a numpy array with shape (2 x 2)."""
        return np.vstack([self.start.to_numpy(), self.end.to_numpy()])

    def intersects_at(self, other: "Line2DBaseGeometry[T]") -> Tuple[Optional[Point2DBaseGeometry[T]], bool, bool]:
        """
        Returns the point at which the two lines intersect and a bool value that indicated if the intersection point is
        within the line segments. Returns None if the lines are parallel.
        See: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        """
        # p and q are the start points. r and s are direction vectors. u and t are scalars
        # p + t r = q + u s
        # we solve for u = (q − p) × r / (r × s)
        # this way the intersection point is q + u s

        p = self.start
        r = self.direction

        q = other.start
        s = other.direction

        rxs = float(np.cross(r.to_numpy(), s.to_numpy()))
        q_pxr = float(np.cross((q - p).to_numpy(), r.to_numpy()))
        q_pxs = float(np.cross((q - p).to_numpy(), s.to_numpy()))

        if rxs == 0 and q_pxr != 0:
            # parallel and non intersecting
            return None, False, False
        elif rxs == 0 and q_pxr == 0:
            # are on the same line
            raise CollinearLinesException("The given lines are collinear!")

        else:
            u = q_pxr / rxs
            t = q_pxs / rxs
            intersection = q + u * s
            return intersection, 0 <= t <= 1, 0 <= u <= 1

    @classmethod
    def from_numpy(cls, points: np.ndarray) -> "Line2DBaseGeometry[T]":
        points = points.reshape(2, 2)
        return Line2DBaseGeometry[T](
            start=Point2DBaseGeometry[T].from_numpy(points[0]), end=Point2DBaseGeometry[T].from_numpy(points[1])
        )


class Line2DGeometry(Line2DBaseGeometry[int]):
    pass


@dataclass
class Polyline2DBaseGeometry(Generic[T]):
    """A polyline made of a collection of 2D Lines

    Args:
        lines: :attr:`~.Polyline2D.lines`

    Attributes:
        lines: Ordered list of :obj:`Line2D` instances
    """

    lines: List[Line2DBaseGeometry[T]]

    @property
    def length(self):
        """Returns the length of the line."""
        return sum([ll.length for ll in self.lines])

    def to_numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 2)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 2))
        elif num_lines == 1:
            return self.lines[0].to_numpy()
        else:
            return np.vstack([self.lines[0].start.to_numpy()] + [ll.end.to_numpy() for ll in self.lines])

    @classmethod
    def from_numpy(cls, points: np.ndarray):
        if points.shape[1] != 2:
            raise ValueError(f"Expected array with shape (N x 2) but got array with shape {points.shape}")

        point_pairs = np.hstack([points[:-1], points[1:]])
        return cls(lines=list(map(Line2DGeometry.from_numpy, point_pairs)))


class Polyline2DGeometry(Polyline2DBaseGeometry[int]):
    pass
