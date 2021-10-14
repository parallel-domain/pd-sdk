from dataclasses import dataclass
from typing import Union

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
    # Customization to cover camera distortion
    # To be removed once DGP 1.x includes distortion params
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    fov: float = 0.0
    fisheye: Union[bool, int] = 0
