from typing import List, Union

import cv2
import numpy as np

from paralleldomain.model.annotation import BoundingBox3D
from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry


class OccupancyGrid:
    def __init__(self, width: float, height: float, offset_x: float, offset_y: float, resolution: float):
        self._resolution = resolution
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._grid = np.zeros((int(height / resolution) + 1, int(width / resolution) + 1))

    def _world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return ((xy - np.array([self._offset_x, self._offset_y])) / self._resolution).astype(int)

    def _grid_to_world(self, xy: np.ndarray) -> np.ndarray:
        return (xy * self._resolution) + np.array([self._offset_x, self._offset_y])

    def occupy_world(self, polygon: np.ndarray):
        """
        Marks the space within the given polygon as occupied.
        Args:
            polygon: A numpy array of a 2D polygon with shape Nx2. Should contain a polygon without self-intersections.

        """
        vertices_cells = self._world_to_grid(xy=polygon[:, :2])
        hull = cv2.convexHull(vertices_cells)
        img = np.zeros_like(self._grid)
        cv2.fillConvexPoly(img, hull, color=(1,))

        self._grid += img

    def is_occupied_world(self, points: np.ndarray) -> np.ndarray:
        """
        Accepts an array of world XY positions and returns an array if points are occupied or not
        Args:
            points: A numpy array of XY points of shape Nx2.
        Returns:
            A numpy array with same length as input, indicating if a world spot is occupied or not.
        """
        xy_cells = self._world_to_grid(xy=points)

        xy_occupied = np.zeros(shape=(xy_cells.shape[0],)).astype(bool)

        xy_cells_in_grid = np.where(
            (xy_cells[:, 0] >= 0)
            & (xy_cells[:, 0] < self._grid.shape[1])
            & (xy_cells[:, 1] >= 0)
            & (xy_cells[:, 1] < self._grid.shape[0])
        )

        xy_occupied[xy_cells_in_grid] = self._grid[xy_cells[xy_cells_in_grid, 1], xy_cells[xy_cells_in_grid, 0]] > 0
        return xy_occupied

    @classmethod
    def from_bounding_boxes_3d(
        cls,
        boxes: Union[List[BoundingBox3DGeometry], List[BoundingBox3D]],
        resolution: float = 0.1,
    ) -> "OccupancyGrid":
        """
        Creates an `OccupancyGrid` from a list of world bounding boxes 3D.
        Args:
            boxes: List of 3D bounding boxes in world coordinates.
            resolution: Grid cell size in `m`. Default: `0.1`

        Returns: OccupancyGrid created from 3D bounding box information

        """

        all_vertices = np.asarray([box.vertices for box in boxes])

        if len(all_vertices) == 0:  # no boxes were provided, so return empty OccupancyGrid
            return cls(width=0, height=0, offset_x=0, offset_y=0, resolution=resolution)

        all_vertices_flat = all_vertices.reshape((-1, 3))
        (min_x, min_y, _), (max_x, max_y, _) = np.amin(all_vertices_flat, axis=0), np.amax(all_vertices_flat, axis=0)

        occupancy_grid = cls(
            width=max_x - min_x, height=max_y - min_y, offset_x=min_x, offset_y=min_y, resolution=resolution
        )

        for i in range(len(all_vertices)):
            vertices = all_vertices[i, ...]
            occupancy_grid.occupy_world(polygon=vertices[:, :2])

        return occupancy_grid
