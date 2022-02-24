from dataclasses import dataclass
from typing import TypeVar

from paralleldomain.model.geometry.polyline_3d import Polyline3DBaseGeometry

T = TypeVar("T", int, float)


@dataclass
class Polygon3DBaseGeometry(Polyline3DBaseGeometry[T]):
    """A closed polygon made a collection of 3D Lines.

    Args:
        lines: :attr:`~.Polygon3DBaseGeometry.lines`
        class_id: :attr:`~.Polygon3DBaseGeometry.class_id`
        instance_id: :attr:`~.Polygon3DBaseGeometry.instance_id`
        attributes: :attr:`~.Polygon3DBaseGeometry.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line3DBaseGeometry` instances
        class_id: Class ID of the polygon. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    ...


class Polygon3DGeometry(Polygon3DBaseGeometry[float]):
    pass
