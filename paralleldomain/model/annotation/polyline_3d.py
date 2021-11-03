from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.annotation.point_3d import Point3D


@dataclass
class Line3D:
    """Represents a 3D Line.

    Args:
        start: :attr:`~.Line3D.start`
        end: :attr:`~.Line3D.end`
        class_id: :attr:`~.Line3D.class_id`
        instance_id: :attr:`~.Line3D.instance_id`
        attributes: :attr:`~.Line3D.attributes`

    Attributes:
        start: the 3D start point of the line in image coordinates
        end: the 3D end point of the line in image coordinates
        directed: whether the line is directed from start to end (if False the line is bi-directional)
        class_id: Class ID of the line. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    start: Point3D
    end: Point3D
    class_id: int
    directed: bool = False
    instance_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self):
        """Returns the start and end coordinates as a numpy array with shape (2 x 3)."""
        return np.vstack([self.start.to_numpy(), self.end.to_numpy()])


@dataclass
class Polyline3D:
    """A polyline made of a collection of 3D Lines

    Args:
        lines: :attr:`~.Polyline3D.lines`
        class_id: :attr:`~.Polyline3D.class_id`
        instance_id: :attr:`~.Polyline3D.instance_id`
        attributes: :attr:`~.Polyline3D.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line3D` instances
        class_id: Class ID of the polyline. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    lines: List[Line3D]
    class_id: int
    instance_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self):
        """Returns all ordered vertices as a numpy array of shape (N x 3)."""
        num_lines = len(self.lines)
        if num_lines == 0:
            return np.empty((0, 3))
        elif num_lines == 1:
            return self.lines[0].to_numpy()
        else:
            return np.vstack([ll.to_numpy()[0] for ll in self.lines[:-1]] + [self.lines[-1].to_numpy()[1]])


@dataclass
class Polylines3D(Annotation):
    """Collection of 3D Polylines

    Args:
        polylines: :attr:`~.Polylines3D.polylines`

    Attributes:
        polylines: Ordered list of :obj:`Polyline3D` instances
    """

    polylines: List[Polyline3D]
