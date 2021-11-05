from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.geometry.point_3d import Point3DGeometry


@dataclass
class Point3D(Point3DGeometry):
    """Represents a 3D Point.

    Args:
        x: :attr:`~.Point3D.x`
        y: :attr:`~.Point3D.y`
        class_id: :attr:`~.Point3D.class_id`
        instance_id: :attr:`~.Point3D.instance_id`
        attributes: :attr:`~.Point3D.attributes`

    Attributes:
        x: coordinate along x-axis in image pixels
        y: coordinate along y-axis in image pixels
        class_id: Class ID of the point. Can be used to lookup more details in :obj:`ClassMap`.
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
class Points3D(Annotation):
    """Collection of 3D Points

    Args:
        points: :attr:`~.Points3D.points`

    Attributes:
        points: Unordered list of :obj:`Point3D` instances
    """

    points: List[Point3D]
