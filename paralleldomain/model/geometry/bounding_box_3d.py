from dataclasses import dataclass

import numpy as np

# _UNIT_BOUNDING_BOX_3D = (CoordinateSystem("FLU") > INTERNAL_COORDINATE_SYSTEM).rotation_matrix @ np.array(
from paralleldomain.utilities.transformation import Transformation

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

    pose: Transformation
    width: float
    height: float
    length: float

    def __repr__(self):
        rep = f"Width: {self.width}, Length: {self.length}, Height: {self.height}, Pose: {self.pose}"
        return rep

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
