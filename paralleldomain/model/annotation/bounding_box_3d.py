from contextlib import suppress
from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List, Optional

import numpy as np

from paralleldomain.model.annotation.common import Annotation, AnnotationPose

# _UNIT_BOUNDING_BOX_3D = (CoordinateSystem("FLU") > INTERNAL_COORDINATE_SYSTEM).rotation_matrix @ np.array(
_UNIT_BOUNDING_BOX_3D = np.array(
    [
        [1, -1, -1],  # FRD
        [1, -1, 1],  # FRU
        [1, 1, 1],  # FLU
        [1, 1, -1],  # FLD
        [-1, -1, -1],  # BRD
        [-1, -1, 1],  # BRU
        [-1, 1, 1],  # BLU
        [-1, 1, -1],  # BLD
    ]
)  # CCW order of points for each face ( [0:4]: Front, [4:8]: Back )


@dataclass
class BoundingBox3D:
    """Represents a 3D Bounding Box geometry.

    Args:
        pose: :attr:`~.BoundingBox3D.pose`
        length: :attr:`~.BoundingBox3D.length`
        width: :attr:`~.BoundingBox3D.width`
        height: :attr:`~.BoundingBox3D.height`
        class_id: :attr:`~.BoundingBox3D.class_id`
        instance_id: :attr:`~.BoundingBox3D.instance_id`
        num_points: :attr:`~.BoundingBox3D.num_points`
        attributes: :attr:`~.BoundingBox3D.attributes`

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
        """Returns volume of 3D Bounding Box in cubic meter."""
        return self.length * self.width * self.height

    @property
    def vertices(self) -> np.ndarray:
        """Returns the 3D vertices of a bounding box.

        Vertices are returned in the following order:

        ::

               5--------6
              /|   top /|
             / |      / |
            1--------2  |
            |  4-----|--7
            | /      | /
            |/       |/ left
            0--------3
              front

        """
        scaled_vertices = np.ones(shape=(_UNIT_BOUNDING_BOX_3D.shape[0], 4))
        scaled_vertices[:, :3] = _UNIT_BOUNDING_BOX_3D * np.array(
            [
                self.length / 2,
                self.width / 2,
                self.height / 2,
            ]
        )

        transformed_vertices = (self.pose @ scaled_vertices.T).T[:, :3]

        return transformed_vertices

    @property
    def edges(self) -> np.ndarray:
        """Returns the 3D edges of a bounding box.

        Edges are returned in order of connecting the vertices in the following order:

        - `[0, 1]`
        - `[1, 2]`
        - `[2, 3]`
        - `[3, 0]`
        - `[4, 5]`
        - `[5, 6]`
        - `[6, 7]`
        - `[7, 4]`
        - `[2, 6]`
        - `[7, 3]`
        - `[1, 5]`
        - `[4, 0]`

        ::

               5--------6
              /|   top /|
             / |      / |
            1--------2  |
            |  4-----|--7
            | /      | /
            |/       |/ left
            0--------3
              front



        """
        vertices = self.vertices
        edges = np.empty(shape=(12, 2, 3))

        edges[0, :, :] = vertices[[0, 1], :]  # FRD -> FRU (0 -> 1)
        edges[1, :, :] = vertices[[1, 2], :]  # FRU -> FLU (1 -> 2)
        edges[2, :, :] = vertices[[2, 3], :]  # FLU -> FLD (2 -> 3)
        edges[3, :, :] = vertices[[3, 0], :]  # FLD -> FRD (3 -> 0)
        edges[4, :, :] = vertices[[4, 5], :]  # BRD -> BRU (4 -> 5)
        edges[5, :, :] = vertices[[5, 6], :]  # BRU -> BLU (5 -> 6)
        edges[6, :, :] = vertices[[6, 7], :]  # BLU -> BLD (6 -> 7)
        edges[7, :, :] = vertices[[7, 4], :]  # BLD -> BRD (7 -> 4)
        edges[8, :, :] = vertices[[2, 6], :]  # FLU -> BLU (2 -> 6)
        edges[9, :, :] = vertices[[7, 3], :]  # BLD -> FLD (7 -> 3)
        edges[10, :, :] = vertices[[1, 5], :]  # FRU -> BRU (1 -> 5)
        edges[11, :, :] = vertices[[4, 0], :]  # BRD -> FRD (5 -> 0)

        return edges

    @property
    def faces(self) -> np.ndarray:
        """Returns the 3D faces of a bounding box.

        Faces are returned in order of connecting the vertices in the following order:

        - `[0, 1, 2, 3]` (front)
        - `[4, 5, 6, 7]` (back)
        - `[3, 2, 6, 7]` (left)
        - `[0, 1, 5, 4]` (right)
        - `[6, 2, 1, 5]` (top)
        - `[7, 3, 0, 4]` (bottom)

        ::

               5--------6
              /|   top /|
             / |      / |
            1--------2  |
            |  4-----|--7
            | /      | /
            |/       |/ left
            0--------3
              front


        """
        vertices = self.vertices
        faces = np.empty(shape=(6, 4, 3))

        faces[0, :, :] = vertices[[0, 1, 2, 3], :]  # front
        faces[1, :, :] = vertices[[4, 5, 6, 7], :]  # back
        faces[2, :, :] = vertices[[3, 2, 6, 7], :]  # left
        faces[3, :, :] = vertices[[0, 1, 5, 4], :]  # right
        faces[4, :, :] = vertices[[6, 2, 1, 5], :]  # up (top)
        faces[5, :, :] = vertices[[7, 3, 0, 4], :]  # down (bottom)

        return faces


@dataclass
class BoundingBoxes3D(Annotation):
    """Collection of 3D Bounding Boxes

    Args:
        boxes: :attr:`~.BoundingBoxes3D.boxes`

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

        source_faces = np.array(
            [
                [source_box.length / 2, 0.0, 0.0, 1.0],
                [-1 * source_box.length / 2, 0.0, 0.0, 1.0],
                [0.0, source_box.width / 2, 0.0, 1.0],
                [0.0, -1.0 * source_box.width / 2, 0.0, 1.0],
                [0.0, 0.0, source_box.height / 2, 1.0],
                [0.0, 0.0, -1 * source_box.height / 2, 1.0],
            ]
        )
        target_faces = np.array(
            [
                [target_box.length / 2, 0.0, 0.0, 1.0],
                [-1 * target_box.length / 2, 0.0, 0.0, 1.0],
                [0.0, target_box.width / 2, 0.0, 1.0],
                [0.0, -1.0 * target_box.width / 2, 0.0, 1.0],
                [0.0, 0.0, target_box.height / 2, 1.0],
                [0.0, 0.0, -1 * target_box.height / 2, 1.0],
            ]
        )
        sensor_frame_faces = source_box.pose @ source_faces.transpose()
        bike_frame_faces = (target_box.pose.inverse @ sensor_frame_faces).transpose()
        max_faces = np.where(np.abs(target_faces) > np.abs(bike_frame_faces), target_faces, bike_frame_faces)
        length = max_faces[0, 0] - max_faces[1, 0]
        width = max_faces[2, 1] - max_faces[3, 1]
        height = max_faces[4, 2] - max_faces[5, 2]
        center = np.array(
            [max_faces[1, 0] + 0.5 * length, max_faces[3, 1] + 0.5 * width, max_faces[5, 2] + 0.5 * height, 1.0]
        )
        translation = target_box.pose @ center
        fused_pose = AnnotationPose(quaternion=target_box.pose.quaternion, translation=translation[:3])
        attributes = target_box.attributes

        result_box = BoundingBox3D(
            pose=fused_pose,
            length=length,
            width=width,
            height=height,
            class_id=target_box.class_id,
            instance_id=target_box.instance_id,
            num_points=(target_box.num_points + source_box.num_points),
            attributes=attributes,
        )

        return result_box
