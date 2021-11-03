from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox2D:
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
        rep = f"x: {self.x}, y: {self.y}, w: {self.width}, h: {self.height}"
        return rep
