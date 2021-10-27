from dataclasses import dataclass
from typing import List

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.annotation.polyline_2d import Polyline2D


@dataclass
class Polygon2D(Polyline2D):
    """A closed polygon made a collection of 2D Lines.

    Args:
        lines: :attr:`~.Polygon2D.lines`
        class_id: :attr:`~.Polygon2D.class_id`
        instance_id: :attr:`~.Polygon2D.instance_id`
        attributes: :attr:`~.Polygon2D.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line2D` instances
        class_id: Class ID of the polygon. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    ...


@dataclass
class Polygons2D(Annotation):
    """Collection of 2D Polygons

    Args:
        polygons: :attr:`~.Polygons2D.polygons`

    Attributes:
        polygons: Ordered list of :obj:`Polygon2D` instances
    """

    polygons: List[Polygon2D]
