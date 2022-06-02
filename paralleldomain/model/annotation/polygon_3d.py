from dataclasses import dataclass
from typing import List

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.annotation.polyline_3d import Polyline3D


@dataclass
class Polygon3D(Polyline3D):
    """A closed polygon made a collection of 3D Lines.

    Args:
        lines: :attr:`paralleldomain.model.annotation.polygon_3d.Polygon3D.lines`
        class_id: :attr:`paralleldomain.model.annotation.polygon_3d.Polygon3D.class_id`
        instance_id: :attr:`paralleldomain.model.annotation.polygon_3d.Polygon3D.instance_id`
        attributes: :attr:`paralleldomain.model.annotation.polygon_3d.Polygon3D.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line3D` instances
        class_id: Class ID of the polygon. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    ...


@dataclass
class Polygons3D(Annotation):
    """Collection of 3D Polygons

    Args:
        polygons: :attr:`paralleldomain.model.annotation.polygon_3d.Polygon3D.polygons`

    Attributes:
        polygons: Ordered list of :obj:`Polygon3D` instances
    """

    polygons: List[Polygon3D]
