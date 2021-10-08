from dataclasses import dataclass
from typing import Union

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


@dataclass
class CameraIntrinsicsDTO(DataClassDictMixin):
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
