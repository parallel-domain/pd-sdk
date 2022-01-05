from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    def to_numpy(self):
        """Returns the start and end coordinates as a numpy array with shape (2 x 2)."""
        return np.vstack([self.start.to_numpy(), self.end.to_numpy()])

    def intersects_at(self, other: "Line2DGeometry") -> Tuple[Optional[Point2DGeometry], bool, bool]:
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
        r = self.end - self.start

        q = other.start
        s = other.end - other.start

        rxs = float(np.cross(r.to_numpy(), s.to_numpy()))
        q_pxr = float(np.cross((q - p).to_numpy(), r.to_numpy()))
        q_pxs = float(np.cross((q - p).to_numpy(), s.to_numpy()))

        if rxs == 0 and q_pxr != 0:
            # parallel and non intersecting
            return None, False, False
        elif rxs == 0 and q_pxr == 0:
            # are on the same line
            # TODO: figure out what to do here
            raise NotImplementedError("Not sure yet what collinear lines should return")
            # len_qp = float(np.linalg.norm((q - p).to_numpy()))
            # len_r = float(np.linalg.norm(r.to_numpy()))
            # r = r.to_numpy() / np.linalg.norm(r.to_numpy())
            # s = s.to_numpy() / np.linalg.norm(s.to_numpy())
            # if np.dot(r, s) == 1:
            #     # point in the same direction. Return start point of line_b
            #     start_copy = other.start * 1
            #     return start_copy, len_r > len_qp
            # else:
            #     # point in the opposite direction. Return end point of line_b
            #     end_copy = other.end * 1
            #     return end_copy, len_r > len_qp

        else:
            u = q_pxr / rxs
            t = q_pxs / rxs
            intersection = q + u * s
            return intersection, 0 <= t <= 1, 0 <= u <= 1


@dataclass
class Polyline2DGeometry:
    """A polyline made of a collection of 2D Lines

    Args:
        lines: :attr:`~.Polyline2D.lines`

    Attributes:
        lines: Ordered list of :obj:`Line2D` instances
    """

    lines: List[Line2DGeometry]

    def to_numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 2)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 2))
        elif num_lines == 1:
            return self.lines[0].to_numpy()
        else:
            return np.vstack([ll.to_numpy()[0] for ll in self.lines[:-1]] + [self.lines[-1].to_numpy()[1]])
