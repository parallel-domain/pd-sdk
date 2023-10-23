import copy
from contextlib import suppress
from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List, Optional

from paralleldomain.model.annotation.common import Annotation
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry, BoundingBox2DGeometry


@dataclass
class BoundingBox2D(BoundingBox2DGeometry):
    """Represents a 2D Bounding Box annotation including geometry

    Args:
        x: :attr:`BoundingBox2D.x`
        y: :attr:`BoundingBox2D.y`
        width: :attr:`BoundingBox2D.width`
        height: :attr:`BoundingBox2D.height`
        class_id: :attr:`BoundingBox2D.class_id`
        instance_id: :attr:`BoundingBox2D.instance_id`
        attributes: :attr:`BoundingBox2D.attributes`

    Attributes:
        x: Top-Left corner of bounding box in image pixels coordinates along x-axis
        y: Top-Left corner of bounding box in image pixels coordinates along y-axis
        width: Width of bounding box in pixels along x-axis
        height: Height of bounding box in pixels along y-axis
        class_id: Class ID of object annotated by bounding box.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`AnnotationTypes.InstanceSegmentation2D` or
            :obj:`AnnotationTypes.InstanceSegmentation3D`.
        attributes: Dictionary of arbitrary object attributes.
    """

    class_id: int
    instance_id: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        rep = f"Class ID: {self.class_id}, Instance ID: {self.instance_id} | {super().__repr__()}"
        return rep

    def __sizeof__(self):
        return getsizeof(self.attributes) + 2 * 8 + super().__sizeof__()  # 2 * 8 bytes ints or floats


@dataclass
class BoundingBoxes2D(Annotation):
    """
    Collection of 2D Bounding Boxes

    Args:
        boxes: :attr:`BoundingBoxes2D.boxes`

    Attributes:
        boxes: Unordered list of :obj:`AnnotationTypes.BoundingBox2D`
    """

    boxes: List[BoundingBox2D]

    def get_box_by_instance_id(self, instance_id: int) -> Optional[BoundingBox2D]:
        """
        Returns the box with matching instance ID.

        Args:
              instance_id: Instance ID of box that should be returned.

        Returns:
              Matching box instance. If none found, returns `None`.
        """
        return next((b for b in self.boxes if b.instance_id == instance_id), None)

    def get_boxes_by_attribute_key(self, attr_key: str) -> List[BoundingBox2D]:
        """
        Returns all boxes having a certain attribute, independent of value.

        Args:
            attr_key: Name of attribute.

        Returns:
            List of box instances that have the specified attribute.
        """
        return [b for b in self.boxes if attr_key in b.attributes]

    def get_boxes_by_attribute_value(self, attr_key: str, attr_value: Any) -> List[BoundingBox2D]:
        """
        Returns all boxes having the specified attribute and value.

        Args:
            attr_key: Name of attribute.
            attr_value: Value of attribute.

        Returns:
            List of box instances that have the specified attribute and value.
        """
        return self.get_boxes_by_attribute_values(attr_key=attr_key, attr_values=[attr_value])

    def get_boxes_by_attribute_values(self, attr_key: str, attr_values: List[Any]) -> List[BoundingBox2D]:
        """
        Returns all boxes having the specified attribute and any of the values.

        Args:
            attr_key: Name of attribute.
            attr_values: Allowed values of attribute.

        Returns:
            List of box instances that have the specified attribute and any of the values.
        """
        with suppress(KeyError):
            result = [b for b in self.boxes if b.attributes[attr_key] in attr_values]
        return result if result is not None else []  # if only KeyError, then result is None

    def get_boxes_by_class_id(self, class_id: int) -> List[BoundingBox2D]:
        """
        Returns all boxes having a the specified class ID.

        Args:
            class_id: Class ID.

        Returns:
            List of box instances that are of the specified class.
        """
        return self.get_boxes_by_class_ids([class_id])

    def get_boxes_by_class_ids(self, class_ids: List[int]) -> List[BoundingBox2D]:
        """
        Returns all boxes having any of the specified class IDs.

        Args:
            class_ids: Class IDs.

        Returns:
            List of box instances that are of any of the specified classes.
        """
        return [b for b in self.boxes if b.class_id in class_ids]

    def __sizeof__(self):
        return sum([getsizeof(b) for b in self.boxes])

    @staticmethod
    def merge_boxes(target_box: BoundingBox2D, source_box: BoundingBox2D) -> BoundingBox2D:
        """
        Merges a 2D Bounding Box into another 2D Bounding Box

        Args:
            target_box: The 2D Bounding Box into which source_box should be merged
            source_box: The 2D Bounding Box which should be merged into target_box

        Attributes:
            2D Bounding Box containing the dimensions of both `target_box` and `source_box` with the attributes of
                `target_box`
        """

        merged_box_geometry = BoundingBox2DGeometry.merge_boxes(target_box=target_box, source_box=source_box)

        source_merged_ids = source_box.attributes.get("merged_instance_ids", set())
        source_merged_ids.add(source_box.instance_id)
        target_box_merged_ids = source_box.attributes.get("merged_instance_ids", set())
        target_box_merged_ids.update(source_merged_ids)

        attributes = copy.deepcopy(target_box.attributes)
        attributes["merged_instance_ids"] = target_box_merged_ids

        return BoundingBox2D(
            x=merged_box_geometry.x,
            y=merged_box_geometry.y,
            width=merged_box_geometry.width,
            height=merged_box_geometry.height,
            class_id=target_box.class_id,
            instance_id=target_box.instance_id,
            attributes=attributes,
        )
