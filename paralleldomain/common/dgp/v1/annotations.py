from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.geometry import PoseDTO


class AnnotationTypeDTO(IntEnum):
    BOUNDING_BOX_2D = 0
    BOUNDING_BOX_3D = 1
    SEMANTIC_SEGMENTATION_2D = 2
    SEMANTIC_SEGMENTATION_3D = 3
    INSTANCE_SEGMENTATION_2D = 4
    INSTANCE_SEGMENTATION_3D = 5
    DEPTH = 6
    SURFACE_NORMALS_2D = 13
    SURFACE_NORMALS_3D = 7
    MOTION_VECTORS_2D = 8
    MOTION_VECTORS_3D = 9
    KEY_POINT_2D = 10
    KEY_LINE_2D = 11
    POLYGON_2D = 12
    AGENT_BEHAVIOR = 14


@dataclass
class BoundingBox2DDTO(DataClassDictMixin):
    x: int
    y: int
    w: int
    h: int


@dataclass
class BoundingBox2DAnnotationDTO(DataClassDictMixin):
    class_id: int
    box: BoundingBox2DDTO
    area: int
    iscrowd: bool
    instance_id: int
    attributes: Dict[str, str]


@dataclass
class BoundingBox3DDTO(DataClassDictMixin):
    pose: PoseDTO
    length: float
    height: float
    occlusion: int
    truncation: float


@dataclass
class BoundingBox3DAnnotationDTO(DataClassDictMixin):
    class_id: int
    box: BoundingBox3DDTO
    instance_id: int
    attributes: Dict[str, str]
    num_points: int


@dataclass
class KeyPoint2DDTO(DataClassDictMixin):
    x: int
    y: int


@dataclass
class KeyPoint2DAnnotationDTO(DataClassDictMixin):
    class_id: int
    point: KeyPoint2DDTO
    attributes: Dict[str, str]
    key: str


@dataclass
class KeyLine2DAnnotationDTO(DataClassDictMixin):
    class_id: int
    vertices: List[KeyPoint2DDTO]
    attributes: Dict[str, str]
    key: str


@dataclass
class PolygonPoint2DDTO(DataClassDictMixin):
    x: int
    y: int


@dataclass
class Polygon2DAnnotationDTO(DataClassDictMixin):
    class_id: int
    vertices: List[PolygonPoint2DDTO]
    attributes: Dict[str, str]


@dataclass
class BoundingBox2DAnnotationsDTO(DataClassDictMixin):
    annotations: List[BoundingBox2DAnnotationDTO]


@dataclass
class BoundingBox3dAnnotationsDTO(DataClassDictMixin):
    annotations: List[BoundingBox3DAnnotationDTO]


@dataclass
class KeyPoint2DAnnotationsDTO(DataClassDictMixin):
    annotations: List[KeyPoint2DAnnotationDTO]


@dataclass
class KeyLine2DAnnotationsDTO(DataClassDictMixin):
    annotations: List[KeyLine2DAnnotationDTO]


@dataclass
class Polygon2DAnnotationsDTO(DataClassDictMixin):
    annotations: List[Polygon2DAnnotationDTO]
