from dataclasses import dataclass

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin


@dataclass
class Vector3DTO(DataClassDictMixin):
    x: float
    y: float
    z: float


@dataclass
class QuaternionDTO(DataClassDictMixin):
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class PoseDTO(DataClassDictMixin):
    translation: Vector3DTO
    rotation: QuaternionDTO


@dataclass_json
@dataclass
class CameraIntrinsicsDTO:
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
