from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.geometry.point_2d import Point2DGeometry


@dataclass
class Point2D(Point2DGeometry):
    """Represents a 2D Point.

    Args:
        x: :attr:`~.Point2D.x`
        y: :attr:`~.Point2D.y`
        class_id: :attr:`~.Point2D.class_id`
        instance_id: :attr:`~.Point2D.instance_id`
        attributes: :attr:`~.Point2D.attributes`

    Attributes:
        x: coordinate along x-axis in image pixels
        y: coordinate along y-axis in image pixels
        class_id: Class ID of the point. Can be used to lookup more details in :obj:`ClassMap`.
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
class Points2D(Annotation):
    """Collection of 2D Points

    Args:
        points: :attr:`~.Points2D.points`

    Attributes:
        points: Unordered list of :obj:`Point2D` instances
    """

    points: List[Point2D]
