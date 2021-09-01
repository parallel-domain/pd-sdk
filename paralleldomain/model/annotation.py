from contextlib import suppress
from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List, Type

import numpy as np

from paralleldomain.model.transformation import Transformation


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


@dataclass
class BoundingBox2D(Annotation):
    """Represents a 2D Bounding Box geometry

    Args:
        x: Top-Left corner in image pixels coordinates along x-axis
        y: Top-Left corner in image pixels coordinates along y-axis
        width: Width of box in pixels along x-axis
        height: Height of box in pixels along y-axis
        class_id: Class ID of annotated object. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.

    """

    x: int  # top left corner (in absolute pixel coordinates)
    y: int  # top left corner (in absolute pixel coordinates)
    width: int  # in absolute pixel coordinates
    height: int  # in absolute pixel coordinates
    class_id: int
    instance_id: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        rep = f"Class ID: {self.class_id}, Instance ID: {self.instance_id}"
        return rep

    def __sizeof__(self):
        return getsizeof(self.attributes) + 6 * 8  # 6 * 8 bytes ints or floats


@dataclass
class BoundingBoxes2D(Annotation):
    boxes: List[BoundingBox2D]

    def get_instance(self, instance_id: int) -> BoundingBox2D:
        return next((b for b in self.boxes if b.instance_id == instance_id), None)

    def get_attribute_key(self, attr_key: str) -> List[BoundingBox2D]:
        return [b for b in self.boxes if attr_key in b.attributes]

    def get_attribute_value(self, attr_key: str, attr_value: Any) -> List[BoundingBox2D]:
        return self.get_attribute_values(attr_key=attr_key, attr_values=[attr_value])

    def get_attribute_values(self, attr_key: str, attr_values: List[Any]) -> List[BoundingBox2D]:
        with suppress(KeyError):
            result = [b for b in self.boxes if b.attributes[attr_key] in attr_values]
        return result if result is not None else []  # if only KeyError, then result is None

    def get_class(self, class_id: int) -> List[BoundingBox2D]:
        return self.get_classes([class_id])

    def get_classes(self, class_ids: List[int]) -> List[BoundingBox2D]:
        return [b for b in self.boxes if b.class_id in class_ids]

    def __sizeof__(self):
        return sum([getsizeof(b) for b in self.boxes])


@dataclass
class BoundingBox3D:
    pose: AnnotationPose
    width: float
    height: float
    length: float
    class_id: int
    instance_id: int
    num_points: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        rep = f"Class ID: {self.class_id}, Instance ID: {self.instance_id}, Pose: {self.pose}"
        return rep

    def __sizeof__(self):
        return getsizeof(self.pose) + getsizeof(self.attributes) + 6 * 8  # 6 * 8 bytes ints or floats

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class BoundingBoxes3D(Annotation):
    boxes: List[BoundingBox3D]

    def get_instance(self, instance_id: int) -> BoundingBox3D:
        return next((b for b in self.boxes if b.instance_id == instance_id), None)

    def get_attribute_key(self, attr_key: str) -> List[BoundingBox3D]:
        return [b for b in self.boxes if attr_key in b.attributes]

    def get_attribute_value(self, attr_key: str, attr_value: Any) -> List[BoundingBox3D]:
        return self.get_attribute_values(attr_key=attr_key, attr_values=[attr_value])

    def get_attribute_values(self, attr_key: str, attr_values: List[Any]) -> List[BoundingBox3D]:
        with suppress(KeyError):
            result = [b for b in self.boxes if b.attributes[attr_key] in attr_values]
        return result if result is not None else []  # if only KeyError, then result is None

    def get_class(self, class_id: int) -> List[BoundingBox3D]:
        return self.get_classes([class_id])

    def get_classes(self, class_ids: List[int]) -> List[BoundingBox3D]:
        return [b for b in self.boxes if b.class_id in class_ids]

    def __sizeof__(self):
        return sum([getsizeof(b) for b in self.boxes])


@dataclass
class SemanticSegmentation2D(Annotation):
    class_ids: np.ndarray

    def get_class(self, class_id: int) -> np.ndarray:
        return self.get_classes(class_ids=[class_id])

    def get_classes(self, class_ids: List[int]) -> np.ndarray:
        return np.isin(self.class_ids, class_ids)

    @property
    def rgb_encoded(self) -> np.ndarray:
        """Converts Class ID mask to RGB representation, with R being the first 8 bit and B being the last 8 bit.

        :return: `np.ndarray`
        """
        return np.concatenate(
            [self.class_ids & 0xFF, self.class_ids >> 8 & 0xFF, self.class_ids >> 16 & 0xFF], axis=-1
        ).astype(np.uint8)

    def __post_init__(self):
        if len(self.class_ids.shape) != 3:
            raise ValueError("Semantic Segmentation class_ids have to have shape (H x W x 1)")
        if self.class_ids.dtype != int:
            raise ValueError(
                f"Semantic Segmentation class_ids has to contain only integers but has {self.class_ids.dtype}!"
            )
        if self.class_ids.shape[2] != 1:
            raise ValueError("Semantic Segmentation class_ids has to have only 1 channel!")

    def __sizeof__(self):
        return getsizeof(self.class_ids)


@dataclass
class InstanceSegmentation2D(Annotation):
    instance_ids: np.ndarray

    def get_instance(self, instance_id: int) -> np.ndarray:
        return self.get_instances(instance_ids=[instance_id])

    def get_instances(self, instance_ids: List[int]) -> np.ndarray:
        return np.isin(self.instance_ids, instance_ids)

    def __sizeof__(self):
        return getsizeof(self.instance_ids)

    @property
    def rgb_encoded(self) -> np.ndarray:
        """Converts Instace ID mask to RGB representation, with R being the first 8 bit and B being the last 8 bit.

        :return: `np.ndarray`
        """
        return np.concatenate(
            [self.instance_ids & 0xFF, self.instance_ids >> 8 & 0xFF, self.instance_ids >> 16 & 0xFF], axis=-1
        ).astype(np.uint8)

    def __post_init__(self):
        if len(self.instance_ids.shape) != 3:
            raise ValueError("Instance Segmentation instance_ids have to have shape (H x W x 1)")
        if self.instance_ids.dtype != int:
            raise ValueError(
                f"Instance Segmentation instance_ids has to contain only integers but has {self.instance_ids.dtype}!"
            )
        if self.instance_ids.shape[2] != 1:
            raise ValueError("Instance Segmentation instance_ids has to have only 1 channel!")


@dataclass
class OpticalFlow(Annotation):
    vectors: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.vectors)


@dataclass
class Depth(Annotation):
    depth: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.depth)


@dataclass
class SemanticSegmentation3D(Annotation):
    class_ids: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.class_ids)


@dataclass
class InstanceSegmentation3D(Annotation):
    instance_ids: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.instance_ids)


AnnotationType = Type[Annotation]


class AnnotationTypes:
    BoundingBoxes2D: Type[BoundingBoxes2D] = BoundingBoxes2D
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D
    InstanceSegmentation2D: Type[InstanceSegmentation2D] = InstanceSegmentation2D
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D
    InstanceSegmentation3D: Type[InstanceSegmentation3D] = InstanceSegmentation3D
    OpticalFlow: Type[OpticalFlow] = OpticalFlow
    Depth: Type[Depth] = Depth
