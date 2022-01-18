from dataclasses import dataclass
from typing import TypeVar

from paralleldomain.model.geometry.polyline_2d import Polyline2DBaseGeometry

T = TypeVar("T", int, float)


@dataclass
class Polygon2DBaseGeometry(Polyline2DBaseGeometry[T]):
    """A closed polygon made a collection of 2D Lines.

    Args:
        lines: :attr:`~.Polygon2DBaseGeometry.lines`
        class_id: :attr:`~.Polygon2DBaseGeometry.class_id`
        instance_id: :attr:`~.Polygon2DBaseGeometry.instance_id`
        attributes: :attr:`~.Polygon2DBaseGeometry.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line2DBaseGeometry` instances
        class_id: Class ID of the polygon. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    ...


class Polygon2DGeometry(Polygon2DBaseGeometry[int]):
    pass
