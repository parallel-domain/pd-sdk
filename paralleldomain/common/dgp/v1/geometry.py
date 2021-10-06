from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Vector3DTO:
    x: float
    y: float
    z: float


@dataclass_json
@dataclass
class QuaternionDTO:
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass_json
@dataclass
class PoseDTO:
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
