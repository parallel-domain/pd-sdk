from contextlib import suppress
from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List, Optional

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry


@dataclass
class BoundingBox3D(BoundingBox3DGeometry):
    """Represents a 3D Bounding Box geometry.

    Args:
        pose: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.pose`
        length: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.length`
        width: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.width`
        height: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.height`
        class_id: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.class_id`
        instance_id: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.instance_id`
        num_points: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.num_points`
        attributes: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBox3D.attributes`

    Attributes:
        pose: 6D Pose for box in 3D sensor space.
        length: Length of box in meter along x-axis.
        width: Width of box in meter along y-axis.
        height: Height of box in meter along z-axis.
        class_id: Class ID of annotated object. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
        num_points: Number of LiDAR points of related :obj:`Sensor`.
        attributes: Dictionary of arbitrary object attributes.
    """

    class_id: int
    instance_id: int
    num_points: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        rep = f"Class ID: {self.class_id}, Instance ID: {self.instance_id} | {super().__repr__()}"
        return rep

    def __sizeof__(self):
        return getsizeof(self.attributes) + 3 * 8 + super().__sizeof__()  # 3 * 8 bytes ints or floats


@dataclass
class BoundingBoxes3D(Annotation):
    """Collection of 3D Bounding Boxes

    Args:
        boxes: :attr:`paralleldomain.model.annotation.bounding_box_3d.BoundingBoxes3D.boxes`

    Attributes:
        boxes: Unordered list of :obj:`BoundingBox3D` instances
    """

    boxes: List[BoundingBox3D]

    def get_box_by_instance_id(self, instance_id: int) -> Optional[BoundingBox3D]:
        """Returns the box with matching instance ID.

        Args:
              instance_id: Instance ID of box that should be returned.

        Returns:
              Matching box instance. If none found, returns `None`.
        """
        return next((b for b in self.boxes if b.instance_id == instance_id), None)

    def get_boxes_by_attribute_key(self, attr_key: str) -> List[BoundingBox3D]:
        """Returns all boxes having a certain attribute, independent of value.

        Args:
            attr_key: Name of attribute.

        Returns:
            List of box instances that have the specified attribute.
        """
        return [b for b in self.boxes if attr_key in b.attributes]

    def get_boxes_by_attribute_value(self, attr_key: str, attr_value: Any) -> List[BoundingBox3D]:
        """Returns all boxes having the specified attribute and value.

        Args:
            attr_key: Name of attribute.
            attr_value: Value of attribute.

        Returns:
            List of box instances that have the specified attribute and value.
        """
        return self.get_boxes_by_attribute_values(attr_key=attr_key, attr_values=[attr_value])

    def get_boxes_by_attribute_values(self, attr_key: str, attr_values: List[Any]) -> List[BoundingBox3D]:
        """Returns all boxes having the specified attribute and any of the values.

        Args:
            attr_key: Name of attribute.
            attr_values: Allowed values of attribute.

        Returns:
            List of box instances that have the specified attribute and any of the values.
        """
        with suppress(KeyError):
            result = [b for b in self.boxes if b.attributes[attr_key] in attr_values]
        return result if result is not None else []  # if only KeyError, then result is None

    def get_boxes_by_class_id(self, class_id: int) -> List[BoundingBox3D]:
        """Returns all boxes having a the specified class ID.

        Args:
            class_id: Class ID.

        Returns:
            List of box instances that are of the specified class.
        """
        return self.get_boxes_by_class_ids([class_id])

    def get_boxes_by_class_ids(self, class_ids: List[int]) -> List[BoundingBox3D]:
        """Returns all boxes having any of the specified class IDs.

        Args:
            class_ids: Class IDs.

        Returns:
            List of box instances that are of any of the specified classes.
        """
        return [b for b in self.boxes if b.class_id in class_ids]

    def __sizeof__(self):
        return sum([getsizeof(b) for b in self.boxes])

    @staticmethod
    def merge_boxes(target_box: BoundingBox3D, source_box: BoundingBox3D) -> BoundingBox3D:
        """
        Takes two 3D boxes as input and merges both into a new box.
        The resulting box has the exact same properties as `target_box`,
        but with extended `source_box` dimensions merged into it.
        """
        merged_box_geometry = BoundingBox3DGeometry.merge_boxes(target_box=target_box, source_box=source_box)

        return BoundingBox3D(
            pose=merged_box_geometry.pose,
            length=merged_box_geometry.length,
            width=merged_box_geometry.width,
            height=merged_box_geometry.height,
            class_id=target_box.class_id,
            instance_id=target_box.instance_id,
            num_points=(target_box.num_points + source_box.num_points),
            attributes=target_box.attributes,
        )
