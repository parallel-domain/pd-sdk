import copy
import typing
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Union

import numpy as np

from paralleldomain.model.geometry.point_2d import Point2DBaseGeometry

T = TypeVar("T", int, float)


@dataclass
class BoundingBox2DBaseGeometry(Generic[T]):
    """Represents a 2D Bounding Box geometry with a generic coordinate precision of either int or float.

    Args:
        x: :attr:`~.BoundingBox2DGeometryBase.x`
        y: :attr:`~.BoundingBox2DGeometryBase.y`
        width: :attr:`~.BoundingBox2DGeometryBase.width`
        height: :attr:`~.BoundingBox2DGeometryBase.height`
        class_id: :attr:`~.BoundingBox2DGeometryBase.class_id`
        instance_id: :attr:`~.BoundingBox2DGeometryBase.instance_id`
        attributes: :attr:`~.BoundingBox2DGeometryBase.attributes`

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

    x: T
    y: T
    width: T
    height: T

    def _ensure_type(self, value: Union[int, float]) -> T:
        try:
            actual_type = typing.get_args(self.__orig_class__)[0]
            if isinstance(actual_type, type):
                return actual_type(value)
            else:
                return type(self.x)(value)
        except AttributeError:
            return type(self.x)(value)

    @property
    def area(self):
        """Returns area of 2D Bounding Box in square pixel."""
        return self.width * self.height

    @property
    def x_min(self) -> T:
        """Returns the minimum x value of the corners of the 2D Bounding Box"""
        return self.x

    @property
    def y_min(self) -> T:
        """Returns the minimum y value of the corners of the 2D Bounding Box"""
        return self.y

    @property
    def x_max(self) -> T:
        """Returns the maximum x value of the corners of the 2D Bounding Box"""
        return self.x + self.width

    @property
    def y_max(self) -> T:
        """Returns the maximum y value of the corners of the 2D Bounding Box"""
        return self.y + self.height

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

    def include_points(
        self, points: Union[List[Point2DBaseGeometry[T]], np.ndarray], inline: bool = False
    ) -> Optional["BoundingBox2DBaseGeometry[T]"]:
        """Extends the dimensions of the box to include the specified point.

        Args:
            points: List of :obj:`Point2DGeometry` which need to be included in updated bounding box.
            inline: When set, do not return a copy of the object but update the current object. Default: `False`.

        Returns:
            A copy of the current object with extended dimensions, if `inline` is set. Otherwise, returns None.

        """

        np_points = points if isinstance(points, np.ndarray) else np.array([[p.x, p.y] for p in points])

        np_points = np.vstack(
            [
                np_points,
                np.array([self.x_min, self.y_min]),
                np.array([self.x_max, self.y_max]),
            ]
        )
        mins = np.amin(np_points, axis=0)
        maxs = np.amax(np_points, axis=0)

        box = copy.deepcopy(self)
        min_x = self._ensure_type(mins[0])
        max_x = self._ensure_type(maxs[0])
        min_y = self._ensure_type(mins[1])
        max_y = self._ensure_type(maxs[1])
        width = max_x - min_x
        height = max_y - min_y
        box.x = min_x
        box.y = min_y
        box.width = width
        box.height = height

        if inline:
            self.x, self.y, self.width, self.height = box.x, box.y, box.width, box.height
            return self
        else:
            return box

    def include_point(
        self, point: Point2DBaseGeometry[T], inline: bool = False
    ) -> Optional["BoundingBox2DBaseGeometry[T]"]:
        """Extends the dimensions of the box to include the specified point.

        Args:
            point: Instance of :obj:`Point2DGeometry` which needs to be included in updated bounding box.
            inline: When set, do not return a copy of the object but update the current object. Default: `False`.

        Returns:
            A copy of the current object with extended dimensions, if `inline` is set. Otherwise, returns None.

        """
        return self.include_points(points=[point], inline=inline)

    def __repr__(self):
        rep = f"x: {self.x}, y: {self.y}, w: {self.width}, h: {self.height}"
        return rep

    @classmethod
    def merge_boxes(
        cls, target_box: "BoundingBox2DBaseGeometry", source_box: "BoundingBox2DBaseGeometry"
    ) -> "BoundingBox2DBaseGeometry[T]":
        """
        Takes two 2D box geometries as input and merges both into a new box geometry.
        The resulting box geometry has dimensions from `target_box` and `source_box`
        merged into it.
        """
        return target_box.include_points(
            points=[Point2DBaseGeometry[T](x=p[0], y=p[1]) for p in source_box.vertices.tolist()]
        )


class BoundingBox2DGeometry(BoundingBox2DBaseGeometry[int]):
    pass
