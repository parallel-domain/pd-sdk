import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from paralleldomain.model.geometry.point_2d import Point2DGeometry


@dataclass
class BoundingBox2DGeometry:
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

    x: float
    y: float
    width: float
    height: float

    @property
    def area(self):
        """Returns area of 2D Bounding Box in square pixel."""
        return self.width * self.height

    @property
    def x_min(self) -> float:
        return self.x

    @property
    def y_min(self) -> float:
        return self.y

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
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

    def include_point(self, point: Point2DGeometry, inline: bool = False) -> Optional["BoundingBox2DGeometry"]:
        """Extends the dimensions of the box to include the specified point.

        Args:
            point: Instance of :obj:`Point2DGeometry` which needs to be included in updated bounding box.
            inline: When set, do not return a copy of the object but update the current object. Default: `False`.

        Returns:
            A copy of the current object with extended dimensions, if `inline` is set. Otherwise, returns None.

        """
        width_from_ul = point.x - self.x
        height_from_ul = point.y - self.y

        x_new, y_new, width_new, height_new = self.x, self.y, self.width, self.height
        if width_from_ul > self.width:
            width_new = width_from_ul
        elif width_from_ul < 0:
            x_new = self.x + width_from_ul

        if height_from_ul > self.height:
            height_new = height_from_ul
        elif height_from_ul < 0:
            y_new = self.y + height_from_ul

        if inline:
            self.x, self.y, self.width, self.height = x_new, y_new, width_new, height_new
        else:
            box_new = copy.deepcopy(self)
            box_new.x, box_new.y, box_new.width, box_new.height = x_new, y_new, width_new, height_new
            return box_new

    def __repr__(self):
        rep = f"x: {self.x}, y: {self.y}, w: {self.width}, h: {self.height}"
        return rep

    @staticmethod
    def merge_boxes(
        target_box: "BoundingBox2DGeometry", source_box: "BoundingBox2DGeometry"
    ) -> "BoundingBox2DGeometry":
        """
        Takes two 2D box geometries as input and merges both into a new box geometry.
        The resulting box geometry has dimensions from `target_box` and `source_box`
        merged into it.
        """
        x_coords = []
        y_coords = []
        for b in [target_box, source_box]:
            x_coords.append(b.x)
            x_coords.append(b.x + b.width)
            y_coords.append(b.y)
            y_coords.append(b.y + b.height)

        x_ul_new = min(x_coords)
        x_width_new = max(x_coords) - x_ul_new
        y_ul_new = min(y_coords)
        y_height_new = max(y_coords) - y_ul_new

        return BoundingBox2DGeometry(
            x=x_ul_new,
            y=y_ul_new,
            width=x_width_new,
            height=y_height_new,
        )
