from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.geometry import CameraIntrinsicsDTO, PoseDTO
from paralleldomain.common.dgp.v1.identifiers import DatumIdDTO
from paralleldomain.common.dgp.v1.image import ImageDTO
from paralleldomain.common.dgp.v1.point_cloud import PointCloudDTO
from paralleldomain.common.dgp.v1.radar_point_cloud import RadarPointCloudDTO


@dataclass_json
@dataclass
class SampleCalibrationDTO:
    names: List[str]
    intrinsics: List[CameraIntrinsicsDTO]
    extrinsics: List[PoseDTO]


class DatumValueDTO(Enum):
    image = ImageDTO
    point_cloud = PointCloudDTO
    radar_point_cloud = RadarPointCloudDTO


@dataclass_json
@dataclass
class DatumDTO:
    id: DatumIdDTO
    key: str
    datum: DatumValueDTO
    next_key: str
    prev_key: str


@dataclass_json
@dataclass
class SampleDTO:
    id: DatumIdDTO
    datum_keys: List[str]
    calibration_key: str
    metadata: Dict[str, AnyDTO]
