from dataclasses import dataclass

from paralleldomain.model.geometry.polyline_3d import Polyline3D


@dataclass
class Polygon3D(Polyline3D):
    """A closed polygon made a collection of 3D Lines.

    Args:
        lines: :attr:`~.Polygon3D.lines`
        class_id: :attr:`~.Polygon3D.class_id`
        instance_id: :attr:`~.Polygon3D.instance_id`
        attributes: :attr:`~.Polygon3D.attributes`

    Attributes:
        lines: Ordered list of :obj:`Line3D` instances
        class_id: Class ID of the polygon. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation3D` or :obj:`InstanceSegmentation3D`.
            If unknown defaults to -1.
        attributes: Dictionary of arbitrary object attributes.
    """

    ...
