from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.geometry.polyline_2d import Line2DGeometry, Polyline2DGeometry


@dataclass
class Line2D(Line2DGeometry):
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

    class_id: int
    directed: bool = False
    instance_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __sizeof__(self):
        return getsizeof(self.attributes) + 3 * 8 + super().__sizeof__()  # 3 * 8 bytes ints or floats


@dataclass
class Polyline2D(Polyline2DGeometry):
    """A polyline made of a collection of 2D Lines

    Args:
        lines: :attr:`~.Polyline2D.lines`
        class_id: :attr:`~.Polyline2D.class_id`
        instance_id: :attr:`~.Polyline2D.instance_id`
        attributes: :attr:`~.Polyline2D.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line2D` instances
        class_id: Class ID of the polyline. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    class_id: int
    instance_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __sizeof__(self):
        return getsizeof(self.attributes) + 2 * 8 + super().__sizeof__()  # 2 * 8 bytes ints or floats


@dataclass
class Polylines2D(Annotation):
    """Collection of 2D Polylines

    Args:
        polylines: :attr:`~.Polylines2D.polylines`

    Attributes:
        polylines: Ordered list of :obj:`Polyline2D` instances
    """

    polylines: List[Polyline2D]
