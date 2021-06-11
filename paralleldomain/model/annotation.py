from __future__ import annotations as ann

import numpy as np
from dataclasses import dataclass, field
from typing import Type, List, Dict, Any, Optional

from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.transformation import Transformation

import numpy as np
from shapely.geometry import Polygon

from paralleldomain.model.transformation import Transformation
from paralleldomain.utilities.image_tools import mask_to_polygons


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


class VirtualAnnotation:
    """
    Use Multiple Inheritance for annotations which are not part of the DGP output,
    but are calculated through SDK.
    """
    ...


@dataclass
class BoundingBox2D(Annotation):
    x: int  # top left corner (in absolute pixel coordinates)
    y: int  # top left corner (in absolute pixel coordinates)
    width: int  # in absolute pixel coordinates
    height: int  # in absolute pixel coordinates
    class_id: int
    instance_id: int
    visibility: float

    def __repr__(self):
        rep = f"Class ID: {self.class_id}, Instance ID: {self.instance_id}"
        return rep


@dataclass
class BoundingBoxes2D(Annotation):
    boxes: List[BoundingBox2D]
    class_map: ClassMap


class ImageMask(Annotation):
    def __init__(self, mask: np.ndarray):
        self._mask = mask

    @property
    def rgb(self) -> np.ndarray:
        return self._mask[:, :, :3]

    @property
    def rgba(self) -> np.ndarray:
        return self._mask


class InstanceSegmentation2D(ImageMask):
    ...


@dataclass
class OpticalFlow(Annotation):
    vectors: np.ndarray


class SemanticSegmentation2D(ImageMask):
    def __init__(self, mask: np.ndarray, class_map: ClassMap):
        super().__init__(mask=mask)
        self.class_map = class_map

    @property
    def labels(self) -> np.ndarray:
        return self._mask[:, :, 0]


class PolygonSegmentation2D(Annotation, VirtualAnnotation):
    def __init__(self, semseg2d: Optional[SemanticSegmentation2D] = None,
                 instanceseg2d: Optional[InstanceSegmentation2D] = None):
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
        polygons = mask_to_polygons(self._semseg2d.labels)
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

        for child_polygon in self._polygons:
            if child_polygon.exterior_points in child_by_parent:
                child_polygon.set_parent(
                    child_by_parent[child_polygon.exterior_points]
                )


@dataclass
class SemanticSegmentation3D(Annotation):
    mask: np.ndarray
    class_map: ClassMap


@dataclass
class InstanceSegmentation3D(Annotation):
    mask: np.ndarray


@dataclass
class BoundingBoxes3D(Annotation):
    boxes: List[BoundingBox3D]
    class_map: ClassMap


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
        rep = f"Class ID: {self.class_id} {self.pose}"
        return rep

    @property
    def size(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height])  # assuming FLU

    @property
    def position(self) -> np.ndarray:
        return self.pose.translation


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
    BoundingBoxes2D: Type[BoundingBoxes2D] = BoundingBoxes2D
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D
    InstanceSegmentation2D: Type[InstanceSegmentation2D] = InstanceSegmentation2D
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D
    InstanceSegmentation3D: Type[InstanceSegmentation3D] = InstanceSegmentation3D
    PolygonSegmentation2D: Type[PolygonSegmentation2D] = PolygonSegmentation2D
    OpticalFlow: Type[OpticalFlow] = OpticalFlow
