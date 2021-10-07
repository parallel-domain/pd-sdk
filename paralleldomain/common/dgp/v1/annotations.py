from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

import numpy as np
from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.geometry import PoseDTO
from paralleldomain.common.dgp.v1.utils import DICT_STR_STR, NP_UINT32


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
    x: np.uint32 = field(metadata=NP_UINT32)
    y: np.uint32 = field(metadata=NP_UINT32)
    w: np.uint32 = field(metadata=NP_UINT32)
    h: np.uint32 = field(metadata=NP_UINT32)


@dataclass
class BoundingBox2DAnnotationDTO(DataClassDictMixin):
    class_id: np.uint32 = field(metadata=NP_UINT32)
    box: BoundingBox2DDTO
    area: np.uint32 = field(metadata=NP_UINT32)
    iscrowd: bool
    instance_id: np.uint32 = field(metadata=NP_UINT32)
    attributes: Dict[str, str] = field(metadata=DICT_STR_STR)


@dataclass
class BoundingBox3DDTO(DataClassDictMixin):
    pose: PoseDTO
    length: float
    height: float
    occlusion: np.uint32 = field(metadata=NP_UINT32)
    truncation: float


@dataclass
class BoundingBox3DAnnotationDTO(DataClassDictMixin):
    class_id: np.uint32 = field(metadata=NP_UINT32)
    box: BoundingBox3DDTO
    instance_id: np.uint32 = field(metadata=NP_UINT32)
    attributes: Dict[str, str] = field(metadata=DICT_STR_STR)
    num_points: np.uint32 = field(metadata=NP_UINT32)


@dataclass
class KeyPoint2DDTO(DataClassDictMixin):
    x: np.uint32 = field(metadata=NP_UINT32)
    y: np.uint32 = field(metadata=NP_UINT32)


@dataclass
class KeyPoint2DAnnotationDTO(DataClassDictMixin):
    class_id: np.uint32 = field(metadata=NP_UINT32)
    point: KeyPoint2DDTO
    attributes: Dict[str, str] = field(metadata=DICT_STR_STR)
    key: str


@dataclass
class KeyLine2DAnnotationDTO(DataClassDictMixin):
    class_id: np.uint32 = field(metadata=NP_UINT32)
    vertices: List[KeyPoint2DDTO]
    attributes: Dict[str, str] = field(metadata=DICT_STR_STR)
    key: str


@dataclass
class PolygonPoint2DDTO(DataClassDictMixin):
    x: np.uint32 = field(metadata=NP_UINT32)
    y: np.uint32 = field(metadata=NP_UINT32)


@dataclass
class Polygon2DAnnotationDTO(DataClassDictMixin):
    class_id: np.uint32 = field(metadata=NP_UINT32)
    vertices: List[PolygonPoint2DDTO]
    attributes: Dict[str, str] = field(metadata=DICT_STR_STR)


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
