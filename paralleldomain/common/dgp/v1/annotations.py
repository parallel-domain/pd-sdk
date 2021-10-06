from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

from dataclasses_json import dataclass_json

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


@dataclass_json
@dataclass
class BoundingBox2DDTO:
    x: int
    y: int
    w: int
    h: int


@dataclass_json
@dataclass
class BoundingBox2DAnnotationDTO:
    class_id: int
    box: BoundingBox2DDTO
    area: int
    iscrowd: bool
    instance_id: int
    attributes: Dict[str, str]


@dataclass_json
@dataclass
class BoundingBox3DDTO:
    pose: PoseDTO
    width: float
    length: float
    height: float
    occlusion: int
    truncation: float


@dataclass_json
@dataclass
class BoundingBox3DAnnotationDTO:
    class_id: int
    box: BoundingBox3DDTO
    instance_id: int
    attributes: Dict[str, str]
    num_points: int


@dataclass_json
@dataclass
class KeyPoint2DDTO:
    x: int
    y: int


@dataclass_json
@dataclass
class KeyPoint2DAnnotationDTO:
    class_id: int
    point: KeyPoint2DDTO
    attributes: Dict[str, str]
    key: str


@dataclass_json
@dataclass
class KeyLine2DAnnotationDTO:
    class_id: int
    vertices: List[KeyPoint2DDTO]
    attributes: Dict[str, str]
    key: str


@dataclass_json
@dataclass
class PolygonPoint2DDTO:
    x: int
    y: int


@dataclass_json
@dataclass
class Polygon2DAnnotationDTO:
    class_id: int
    vertices: List[PolygonPoint2DDTO]
    attributes: Dict[str, str]


@dataclass_json
@dataclass
class BoundingBox2DAnnotationsDTO:
    annotations: List[BoundingBox2DAnnotationDTO]


@dataclass_json
@dataclass
class BoundingBox3dAnnotationsDTO:
    annotations: List[BoundingBox3DAnnotationDTO]


@dataclass_json
@dataclass
class KeyPoint2DAnnotationsDTO:
    annotations: List[KeyPoint2DAnnotationDTO]


@dataclass_json
@dataclass
class KeyLine2DAnnotationsDTO:
    annotations: List[KeyLine2DAnnotationDTO]


@dataclass_json
@dataclass
class Polygon2DAnnotationsDTO:
    annotations: List[Polygon2DAnnotationDTO]
