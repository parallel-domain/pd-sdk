from contextlib import suppress
from dataclasses import dataclass, field
from itertools import filterfalse
from sys import getsizeof
from typing import Any, Dict, List, Type

import numpy as np

from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
from paralleldomain.model.transformation import Transformation


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


@dataclass
class BoundingBox2D(Annotation):
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
    class_map: ClassMap

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

    def update_instance(self, instance_id: int, box: BoundingBox2D) -> None:
        self.boxes = [b if b.instance_id != instance_id else box for b in self.boxes]

    def remove_instance(self, instance_id: int) -> None:
        self.remove_instances(instance_ids=[instance_id])

    def remove_instances(self, instance_ids: List[int]) -> None:
        self.boxes = list(filterfalse(lambda b: b.instance_id in instance_ids, self.boxes))

    def remove_class(self, class_id: int) -> None:
        self.remove_classes([class_id])

    def remove_classes(self, class_ids: List[int]) -> None:
        self.boxes = list(filterfalse(lambda b: b.class_id in class_ids, self.boxes))

    def merge_instances(self, target_id: int, source_id: int, replace_target: bool = True) -> BoundingBox2D:
        # merges the "source" box into the "target" box
        # does not remove "source" box -> needs to be called manually when desired
        x_coords = []
        y_coords = []
        target = self.get_instance(target_id)
        source = self.get_instance(source_id)
        for b in [source, target]:
            x_coords.append(b.x)
            x_coords.append(b.x + b.width)
            y_coords.append(b.y)
            y_coords.append(b.y + b.height)

        x_ul_new = min(x_coords)
        x_width_new = max(x_coords) - x_ul_new
        y_ul_new = min(y_coords)
        y_height_new = max(y_coords) - y_ul_new

        merged = BoundingBox2D(
            x=x_ul_new,
            y=y_ul_new,
            width=x_width_new,
            height=y_height_new,
            class_id=target.class_id,
            instance_id=target.instance_id,
            attributes=target.attributes,
        )

        if replace_target:
            self.update_instance(target_id, merged)

        return merged

    def update_classes(self, class_id_map: ClassIdMap, class_label_map: ClassMap) -> None:
        for b in self.boxes:
            b.class_id = class_id_map[b.class_id]
        self.class_map = class_label_map

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
    class_map: ClassMap

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

    def update_instance(self, instance_id: int, box: BoundingBox3D) -> None:
        self.boxes = [b if b.instance_id != instance_id else box for b in self.boxes]

    def remove_instance(self, instance_id: int) -> None:
        self.remove_instances([instance_id])

    def remove_instances(self, instance_ids: List[int]) -> None:
        self.boxes = list(filterfalse(lambda b: b.instance_id in instance_ids, self.boxes))

    def remove_class(self, class_id: int) -> None:
        self.remove_classes([class_id])

    def remove_classes(self, class_ids: List[int]) -> None:
        self.boxes = list(filterfalse(lambda b: b.class_id in class_ids, self.boxes))

    def merge_instances(self, target_id: int, source_id: int, replace_target: bool = True) -> BoundingBox3D:
        # merges the "source" box into the "target" box
        # does not remove "source" box -> needs to be called manually when desired
        target = self.get_instance(target_id)
        source = self.get_instance(source_id)

        source_faces = np.array(
            [
                [source.length / 2, 0.0, 0.0, 1.0],
                [-1 * source.length / 2, 0.0, 0.0, 1.0],
                [0.0, source.width / 2, 0.0, 1.0],
                [0.0, -1.0 * source.width / 2, 0.0, 1.0],
                [0.0, 0.0, source.height / 2, 1.0],
                [0.0, 0.0, -1 * source.height / 2, 1.0],
            ]
        )
        target_faces = np.array(
            [
                [target.length / 2, 0.0, 0.0, 1.0],
                [-1 * target.length / 2, 0.0, 0.0, 1.0],
                [0.0, target.width / 2, 0.0, 1.0],
                [0.0, -1.0 * target.width / 2, 0.0, 1.0],
                [0.0, 0.0, target.height / 2, 1.0],
                [0.0, 0.0, -1 * target.height / 2, 1.0],
            ]
        )
        sensor_frame_faces = source.pose @ source_faces.transpose()
        bike_frame_faces = (target.pose.inverse @ sensor_frame_faces).transpose()
        max_faces = np.where(np.abs(target_faces) > np.abs(bike_frame_faces), target_faces, bike_frame_faces)
        length = max_faces[0, 0] - max_faces[1, 0]
        width = max_faces[2, 1] - max_faces[3, 1]
        height = max_faces[4, 2] - max_faces[5, 2]
        center = np.array(
            [max_faces[1, 0] + 0.5 * length, max_faces[3, 1] + 0.5 * width, max_faces[5, 2] + 0.5 * height, 1.0]
        )
        translation = target.pose @ center
        fused_pose = AnnotationPose(quaternion=target.pose.quaternion, translation=translation[:3])
        attributes = target.attributes
        # attributes.update(source.attributes)
        merged = BoundingBox3D(
            pose=fused_pose,
            length=length,
            width=width,
            height=height,
            class_id=target.class_id,
            instance_id=target.instance_id,
            num_points=(target.num_points + source.num_points),
            attributes=attributes,
        )

        if replace_target:
            self.update_instance(target_id, merged)

        return merged

    def update_classes(self, class_id_map: ClassIdMap, class_label_map: ClassMap) -> None:
        for b in self.boxes:
            b.class_id = class_id_map[b.class_id]
        self.class_map = class_label_map

    def __sizeof__(self):
        return sum([getsizeof(b) for b in self.boxes])


@dataclass
class SemanticSegmentation2D(Annotation):
    class_ids: np.ndarray
    class_map: ClassMap

    def get_class(self, class_id: int) -> np.ndarray:
        return self.get_classes(class_ids=[class_id])

    def get_classes(self, class_ids: List[int]) -> np.ndarray:
        return np.isin(self.class_ids, class_ids)

    def update(self, mask: np.ndarray, class_id: int) -> None:
        self.class_ids[mask] = class_id

    def update_classes(self, class_id_map: ClassIdMap, class_label_map: ClassMap) -> None:
        self.class_ids = class_id_map[self.class_ids]
        self.class_map = class_label_map

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
        if self.class_ids.dtype != np.int:
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

    def remove_instance(self, instance_id: int) -> None:
        self.remove_instances([instance_id])

    def remove_instances(self, instance_ids: List[int]) -> None:
        self.update(mask=np.where(np.isin(self.instance_ids, instance_ids)), instance_id=0)

    def update(self, mask: np.ndarray, instance_id: int) -> None:
        self.instance_ids[mask] = instance_id

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
        if self.instance_ids.dtype != np.int:
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
    class_map: ClassMap

    def update_classes(self, class_id_map: ClassIdMap, class_label_map: ClassMap) -> None:
        self.class_ids = class_id_map[self.class_ids]
        self.class_map = class_label_map

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
