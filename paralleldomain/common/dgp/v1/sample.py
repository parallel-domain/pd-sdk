from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.file_datum import FileDatumDTO
from paralleldomain.common.dgp.v1.geometry import CameraIntrinsicsDTO, PoseDTO
from paralleldomain.common.dgp.v1.identifiers import DatumIdDTO
from paralleldomain.common.dgp.v1.image import ImageDTO
from paralleldomain.common.dgp.v1.point_cloud import PointCloudDTO
from paralleldomain.common.dgp.v1.radar_point_cloud import RadarPointCloudDTO
from paralleldomain.common.dgp.v1.utils import SkipNoneMixin


@dataclass_json
@dataclass
class SampleCalibrationDTO:
    names: List[str]
    intrinsics: List[CameraIntrinsicsDTO]
    extrinsics: List[PoseDTO]


@dataclass_json
@dataclass
class DatumValueDTO(SkipNoneMixin):
    image: Optional[ImageDTO] = None
    point_cloud: Optional[PointCloudDTO] = None
    file_datum: Optional[FileDatumDTO] = None
    radar_point_cloud: Optional[RadarPointCloudDTO] = None


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
    metadata: Dict[str, Any]
