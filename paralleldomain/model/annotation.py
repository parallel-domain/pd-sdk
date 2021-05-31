from __future__ import annotations as ann

from dataclasses import dataclass
from typing import Type, List, Dict, Any

from paralleldomain.model.transformation import Transformation

from shapely.geometry import Polygon
import rasterio.features

import numpy as np

from paralleldomain.utilities.image_tools import mask_to_polygons


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


class BoundingBox2D(Annotation):
    ...


@dataclass
class SemanticSegmentation2D(Annotation):
    mask: np.ndarray
    _polygons: List[Polygon2D] = None

    @property
    def rgb(self):
        return self.mask[:, :, :3]

    @property
    def rgba(self):
        return self.mask

    @property
    def labels(self):
        return self.mask[:, :, 0]

    @property
    def polygons(self):
        if self._polygons is None:
            self._mask_to_polygons()

        return self._polygons

    def _mask_to_polygons(self):
        polygons = mask_to_polygons(self.labels)
        self._polygons = [Polygon2D.from_rasterio_polygon(p[0]) for p in polygons]


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

    @property
    def area(self):
        return self._polygon.area

    @property
    def exterior_points(self):
        return np.asarray(self._polygon.exterior.coords)

    @property
    def interior_points(self):
        return [np.asarray(ip.coords) for ip in self._polygon.interiors]

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
