from contextlib import suppress
from dataclasses import dataclass, field
from sys import getsizeof
from typing import Any, Dict, List, Optional, Type

import numpy as np

from paralleldomain.utilities.mask import boolean_mask_by_value, boolean_mask_by_values, encode_int32_as_rgb8
from paralleldomain.utilities.transformation import Transformation

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


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


@dataclass
class BoundingBox2D(Annotation):
    """Represents a 2D Bounding Box geometry.

    Args:
        x: :attr:`~.BoundingBox2D.x`
        y: :attr:`~.BoundingBox2D.y`
        width: :attr:`~.BoundingBox2D.width`
        height: :attr:`~.BoundingBox2D.height`
        class_id: :attr:`~.BoundingBox2D.class_id`
        instance_id: :attr:`~.BoundingBox2D.instance_id`
        attributes: :attr:`~.BoundingBox2D.attributes`

    Attributes:
        x: Top-Left corner in image pixels coordinates along x-axis
        y: Top-Left corner in image pixels coordinates along y-axis
        width: Width of box in pixel along x-axis
        height: Height of box in pixel along y-axis
        class_id: Class ID of annotated object. Can be used to lookup more details in :obj:`ClassMap`.
        instance_id: Instance ID of annotated object. Can be used to cross-reference with
            other instance annotation types, e.g., :obj:`InstanceSegmentation2D` or :obj:`InstanceSegmentation3D`.
        attributes: Dictionary of arbitrary object attributes.
    """

    x: int
    y: int
    width: int
    height: int
    class_id: int
    instance_id: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def area(self):
        """Returns area of 2D Bounding Box in square pixel."""
        return self.width * self.height

    @property
    def vertices(self) -> np.ndarray:
        """Returns the 2D vertices of a bounding box.

        Vertices are returned in the following order:

        ::

            0--------1
            |        |
            |        | right
            |        |
            3--------2
              bottom

        """

        vertices = np.array(
            [
                [self.x, self.y],
                [self.x + self.width, self.y],
                [self.x + self.width, self.y + self.height],
                [self.x, self.y + self.height],
            ]
        )

        return vertices

    @property
    def edges(self) -> np.ndarray:
        """Returns the 2D edges of a bounding box.

        Edges are returned in order of connecting the vertices in the following order:

        - `[0, 1]`
        - `[1, 2]`
        - `[2, 3]`
        - `[3, 0]`

        ::

            0--------1
            |        |
            |        | right
            |        |
            3--------2
              bottom



        """
        vertices = self.vertices
        edges = np.empty(shape=(4, 2, 2))

        edges[0, :, :] = vertices[[0, 1], :]  # UL -> UR (0 -> 1)
        edges[1, :, :] = vertices[[1, 2], :]  # UR -> LR (1 -> 2)
        edges[2, :, :] = vertices[[2, 3], :]  # LR -> LL (2 -> 3)
        edges[3, :, :] = vertices[[3, 0], :]  # LL -> UL (3 -> 0)

        return edges

    def __repr__(self):
        rep = f"Class ID: {self.class_id}, Instance ID: {self.instance_id}"
        return rep

    def __sizeof__(self):
        return getsizeof(self.attributes) + 6 * 8  # 6 * 8 bytes ints or floats


@dataclass
class BoundingBoxes2D(Annotation):
    """Collection of 2D Bounding Boxes.

    Args:
        boxes: :attr:`~.BoundingBoxes2D.boxes`

    Attributes:
        boxes: Unordered list of :obj:`BoundingBox2D` instances
    """

    boxes: List[BoundingBox2D]

    def get_box_by_instance_id(self, instance_id: int) -> Optional[BoundingBox2D]:
        """Returns the box with matching instance ID.

        Args:
              instance_id: Instance ID of box that should be returned.

        Returns:
              Matching box instance. If none found, returns `None`.
        """
        return next((b for b in self.boxes if b.instance_id == instance_id), None)

    def get_boxes_by_attribute_key(self, attr_key: str) -> List[BoundingBox2D]:
        """Returns all boxes having a certain attribute, independent of value.

        Args:
            attr_key: Name of attribute.

        Returns:
            List of box instances that have the specified attribute.
        """
        return [b for b in self.boxes if attr_key in b.attributes]

    def get_boxes_by_attribute_value(self, attr_key: str, attr_value: Any) -> List[BoundingBox2D]:
        """Returns all boxes having the specified attribute and value.

        Args:
            attr_key: Name of attribute.
            attr_value: Value of attribute.

        Returns:
            List of box instances that have the specified attribute and value.
        """
        return self.get_boxes_by_attribute_values(attr_key=attr_key, attr_values=[attr_value])

    def get_boxes_by_attribute_values(self, attr_key: str, attr_values: List[Any]) -> List[BoundingBox2D]:
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

    def get_boxes_by_class_id(self, class_id: int) -> List[BoundingBox2D]:
        """Returns all boxes having a the specified class ID.

        Args:
            class_id: Class ID.

        Returns:
            List of box instances that are of the specified class.
        """
        return self.get_boxes_by_class_ids([class_id])

    def get_boxes_by_class_ids(self, class_ids: List[int]) -> List[BoundingBox2D]:
        """Returns all boxes having any of the specified class IDs.

        Args:
            class_ids: Class IDs.

        Returns:
            List of box instances that are of any of the specified classes.
        """
        return [b for b in self.boxes if b.class_id in class_ids]

    def __sizeof__(self):
        return sum([getsizeof(b) for b in self.boxes])


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


@dataclass
class SemanticSegmentation2D(Annotation):
    """Represents a 2D Semantic Segmentation mask for a camera image.

    Args:
        class_ids: :attr:`~.SemanticSegmentation2D.class_ids`

    Attributes:
        class_ids: Matrix of shape `(H x W x 1)`, where `H` is height and `W` is width of corresponding camera image.
            The third axis contains the class ID for each pixel as `int`.
    """

    class_ids: np.ndarray

    def get_class_mask(self, class_id: int) -> np.ndarray:
        """Returns a `bool` mask where class is present.

        Args:
            class_id: ID of class to be masked

        Returns:
            Mask of same shape as :py:attr:`~class_ids` and `bool` values.
            `True` where pixel matches class, `False` where it doesn't.
        """
        return boolean_mask_by_value(mask=self.class_ids, value=class_id)

    def get_classes_mask(self, class_ids: List[int]) -> np.ndarray:
        """Returns a `bool` mask where classes are present.

        Args:
            class_ids: IDs of classes to be masked

        Returns:
            Mask of same shape as `class_ids` and `bool` values.
            `True` where pixel matches one of the classes, `False` where it doesn't.
        """
        return boolean_mask_by_values(mask=self.class_ids, values=class_ids)

    @property
    def rgb_encoded(self) -> np.ndarray:
        """Outputs :attr:`~.SemanticSegmentation.class_ids` mask as RGB-encoded image matrix with shape `(H x W x 3)`,
        with `R` (index: 0) being the lowest and `B` (index: 2) being the highest 8 bit."""
        return encode_int32_as_rgb8(mask=self.class_ids)

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
    """Represents a 2D Instance Segmentation mask for a camera image.

    Args:
        instance_ids: :attr:`~.SemanticSegmentation2D.instance_ids`

    Attributes:
        instance_ids: Matrix of shape `(H x W x 1)`, where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the instance ID for each pixel as `int`.
    """

    instance_ids: np.ndarray

    def get_instance(self, instance_id: int) -> np.ndarray:
        """Returns a `bool` mask where instance is present.

        Args:
            instance_id: ID of instance to be masked

        Returns:
            Mask of same shape as :py:attr:`~class_ids` and `bool` values.
            `True` where pixel matches instance, `False` where it doesn't.
        """
        return boolean_mask_by_value(mask=self.instance_ids, value=instance_id)

    def get_instances(self, instance_ids: List[int]) -> np.ndarray:
        """Returns a `bool` mask where instances are present.

        Args:
            instance_ids: IDs of instances to be masked

        Returns:
            Mask of same shape as `class_ids` and `bool` values.
            `True` where pixel matches one of the instances, `False` where it doesn't.
        """
        return boolean_mask_by_values(mask=self.instance_ids, values=instance_ids)

    def __sizeof__(self):
        return getsizeof(self.instance_ids)

    @property
    def rgb_encoded(self) -> np.ndarray:
        """Outputs :attr:`~.InstanceSegmentation.instance_ids` mask as RGB matrix with shape `(H x W x 3)`,
        with `R` being the lowest and `B` being the highest 8 bit."""
        return encode_int32_as_rgb8(mask=self.instance_ids)

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
    """Represents an Optical Flow mask for a camera image.

    Args:
        vectors: :attr:`~.OpticalFlow.vectors`

    Attributes:
        vectors: Matrix of shape `(H X W x 2)`, , where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the x and y offset to the pixels coordinate on the next image.

    Example:
        Using the Optical Flow vector mask in combination with :attr:`.ImageData.coordinates` allows for a
        fast retrieval of absolute pixel coordinates.
        ::

            camera_frame: SensorFrame = ...  # get any camera's SensorFrame

            flow = camera_frame.get_annotations(AnnotationTypes.OpticalFlow)
            rgb = camera_frame.image.rgb
            next_image = np.zeros_like(rgb)
            coordinates = camera_frame.image.coordinates
            next_frame_coords = coordinates + flow.vectors

            for y in range(rgb.shape[0]):
                for x in range(rgb.shape[1]):
                    next_coord = next_frame_coords[y, x]
                    if 0 <= next_coord[0] < rgb.shape[0] and 0 <= next_coord[1] < rgb.shape[1]:
                        next_image[next_coord[0], next_coord[1], :] = rgb[y, x, :]

            import cv2
            cv2.imshow("window_name", cv2.cvtColor(
                    src=next_image,
                    code=cv2.COLOR_RGBA2BGRA,
            ))
            cv2.waitKey()
    """

    vectors: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.vectors)


@dataclass
class Depth(Annotation):
    """Represents a Depth mask for a camera image.



    Args:
        depth: :attr:`~.Depth.depth`

    Attributes:
        depth: Matrix of shape `(H X W x 1)`, , where `H` is the height and `W` is the width of corresponding
            camera image. The third axis contains the depth distance for each pixel as `int` in meter.

    """

    depth: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.depth)


@dataclass
class SemanticSegmentation3D(Annotation):
    """Represents a 3D Instance Segmentation mask for a point cloud.

    Args:
        class_ids: :attr:`~.SemanticSegmentation3D.class_ids`

    Attributes:
        class_ids: Matrix of shape `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the class ID for each point as `int`.
    """

    class_ids: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.class_ids)


@dataclass
class InstanceSegmentation3D(Annotation):
    """Represents a 3D Instance Segmentation mask for a point cloud.

    Args:
        instance_ids: :attr:`~.InstanceSegmentation3D.instance_ids`

    Attributes:
        instance_ids: 2D Matrix of size `(N x 1)`, where `N` is the length of the corresponding point cloud.
            The second axis contains the instance ID for each point as `int`.
    """

    instance_ids: np.ndarray

    def __sizeof__(self):
        return getsizeof(self.instance_ids)


@dataclass
class SurfaceNormals3D(Annotation):
    """
    Not Implemented yet!
    """

    ...


@dataclass
class SurfaceNormals2D(Annotation):
    """
    Not Implemented yet!
    """

    ...


@dataclass
class SceneFlow(Annotation):
    """
    Not Implemented yet!
    """

    ...


@dataclass
class MaterialProperties2D(Annotation):
    """
    Not Implemented yet!
    """

    ...


@dataclass
class Albedo2D(Annotation):
    """
    Not Implemented yet!
    """

    ...


AnnotationType = Type[Annotation]


class AnnotationTypes:
    """Allows to get type-safe access to annotation type related information, e.g., annotation data or class maps.

    Attributes:
        BoundingBoxes2D
        BoundingBoxes3D
        SemanticSegmentation2D
        InstanceSegmentation2D
        SemanticSegmentation3D
        InstanceSegmentation3D
        OpticalFlow
        Depth

    Examples:
        Access 2D Bounding Box annotations for a camera frame:
        ::

            camera_frame: SensorFrame = ...  # get any camera's SensorFrame

            from paralleldomain.model.annotation import AnnotationTypes

            boxes_2d = camera_frame.get_annotations(AnnotationTypes.BoundingBoxes2D)
            for b in boxes_2d.boxes:
                print(b.class_id, b.instance_id)

        Access class map for an annotation type in a scene:
        ::

            scene: Scene = ...  # get a Scene instance

            from paralleldomain.model.annotation import AnnotationTypes

            class_map = scene.get_class_map(AnnotationTypes.SemanticSegmentation2D)
            for id, class_detail in class_map.items():
                print(id, class_detail.name)
    """

    BoundingBoxes2D: Type[BoundingBoxes2D] = BoundingBoxes2D
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D
    InstanceSegmentation2D: Type[InstanceSegmentation2D] = InstanceSegmentation2D
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D
    InstanceSegmentation3D: Type[InstanceSegmentation3D] = InstanceSegmentation3D
    OpticalFlow: Type[OpticalFlow] = OpticalFlow
    Depth: Type[Depth] = Depth
    SurfaceNormals3D: Type[SurfaceNormals3D] = SurfaceNormals3D
    SurfaceNormals2D: Type[SurfaceNormals2D] = SurfaceNormals2D
    SceneFlow: Type[SceneFlow] = SceneFlow
    MaterialProperties2D: Type[MaterialProperties2D] = MaterialProperties2D
    Albedo2D: Type[Albedo2D] = Albedo2D
