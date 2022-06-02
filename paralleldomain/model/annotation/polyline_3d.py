from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.geometry.polyline_3d import Line3DGeometry, Polyline3DGeometry


@dataclass
class Line3D(Line3DGeometry):
    """Represents a 3D Line.

    Args:
        start: :attr:`paralleldomain.model.annotation.polyline_3d.Line3D.start`
        end: :attr:`paralleldomain.model.annotation.polyline_3d.Line3D.end`
        class_id: :attr:`paralleldomain.model.annotation.polyline_3d.Line3D.class_id`
        instance_id: :attr:`paralleldomain.model.annotation.polyline_3d.Line3D.instance_id`
        attributes: :attr:`paralleldomain.model.annotation.polyline_3d.Line3D.attributes`

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

    class_id: int
    directed: bool = False
    instance_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __sizeof__(self):
        return getsizeof(self.attributes) + 2 * 8 + super().__sizeof__()  # 2 * 8 bytes ints or floats


@dataclass
class Polyline3D(Polyline3DGeometry):
    """A polyline made of a collection of 3D Lines

    Args:
        lines: :attr:`paralleldomain.model.annotation.polyline_3d.Polyline3D.lines`
        class_id: :attr:`paralleldomain.model.annotation.polyline_3d.Polyline3D.class_id`
        instance_id: :attr:`paralleldomain.model.annotation.polyline_3d.Polyline3D.instance_id`
        attributes: :attr:`paralleldomain.model.annotation.polyline_3d.Polyline3D.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line3D` instances
        class_id: Class ID of the polyline. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    class_id: int
    instance_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __sizeof__(self):
        return getsizeof(self.attributes) + 2 * 8 + super().__sizeof__()  # 2 * 8 bytes ints or floats


@dataclass
class Polylines3D(Annotation):
    """Collection of 3D Polylines

    Args:
        polylines: :attr:`paralleldomain.model.annotation.polyline_3d.Polylines3D.polylines`

    Attributes:
        polylines: Ordered list of :obj:`Polyline3D` instances
    """

    polylines: List[Polyline3D]
