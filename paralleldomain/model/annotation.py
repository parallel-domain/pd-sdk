from __future__ import annotations as ann

from dataclasses import dataclass
from typing import Type, List, Dict, Any

import numpy as np
from shapely.geometry import Polygon

from paralleldomain.model.transformation import Transformation
from paralleldomain.utilities.image_tools import mask_to_polygons


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


class BoundingBox2D(Annotation):
    ...


class InstanceSegmentation2D(Annotation):
    ...


class SemanticSegmentation2D(Annotation):
    def __init__(self, mask: np.ndarray):
        self._mask = mask

    @property
    def rgb(self) -> np.ndarray:
        return self._mask[:, :, :3]

    @property
    def rgba(self) -> np.ndarray:
        return self._mask

    @property
    def labels(self) -> np.ndarray:
        return self._mask[:, :, 0]


class PolygonSegmentation2D(Annotation):
    def __init__(self, semseg2d: SemanticSegmentation2D = None, instanceseg2d: InstanceSegmentation2D = None):
        self._semseg2d = semseg2d
        self._instanceseg2d = instanceseg2d
        self._polygons = None

    @property
    def polygons(self) -> List[Polygon2D]:
        if self._polygons is None:
            self._mask_to_polygons()
            self._build_polygon_tree()

        return self._polygons

    def _mask_to_polygons(self) -> None:
        polygons = mask_to_polygons(self.labels)
        self._polygons = [Polygon2D.from_rasterio_polygon(p[0]) for p in polygons]

    def _build_polygon_tree(self) -> None:
        """ Compare LinearRings on tuple-level so it is hashable for performance

        :return:
        """
        child_by_parent = {
            p_interior: p
            for p in self._polygons
            for p_interior in p.interior_points
        }

        _ = [c.set_parent(child_by_parent[c.exterior_points]) for c in self._polygons if
             tuple(c.exterior_points) in child_by_parent]


@dataclass
class SemanticSegmentation3D(Annotation):
    mask: np.ndarray


@dataclass
class BoundingBoxes3D(Annotation):
    boxes: List[BoundingBox3D]


@dataclass
class BoundingBox3D:
    pose: AnnotationPose
    width: float
    height: float
    length: float
    class_id: int
    class_name: str
    instance_id: int
    num_points: int

    def __repr__(self):
        rep = f"Class ID: {self.class_id} {self.pose}"
        return rep


class Polygon2D:
    def __init__(self, polygon: Polygon):
        self._polygon = polygon
        self._parent = None

    @property
    def area(self):
        return abs(self._polygon.area)

    @property
    def z_index(self):
        if self._parent == None:
            return 0
        else:
            return self._parent.z_index + 1

    @property
    def has_children(self):
        return len(self._polygon.interiors) > 0

    @property
    def exterior_points(self):
        return tuple(self._polygon.exterior.coords)

    @property
    def interior_points(self):
        return [tuple(ip.coords) for ip in self._polygon.interiors]

    def set_parent(self, parent: Polygon2D):
        self._parent = parent

    @staticmethod
    def from_rasterio_polygon(polygon_dict: Dict[str: Any]):
        coordinates = polygon_dict["coordinates"]
        polygon = Polygon(
            coordinates[0],
            holes=coordinates[1:]
        )

        return Polygon2D(polygon=polygon)


AnnotationType = Type[Annotation]


class AnnotationTypes:
    BoundingBox2D: Type[BoundingBox2D] = BoundingBox2D
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D
    PolygonSegmentation2D: Type[PolygonSegmentation2D] = PolygonSegmentation2D
