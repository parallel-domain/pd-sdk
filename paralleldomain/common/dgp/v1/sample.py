from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin
from mashumaro.config import TO_DICT_ADD_OMIT_NONE_FLAG, BaseConfig

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.file_datum import FileDatumDTO
from paralleldomain.common.dgp.v1.geometry import CameraIntrinsicsDTO, PoseDTO
from paralleldomain.common.dgp.v1.identifiers import DatumIdDTO
from paralleldomain.common.dgp.v1.image import ImageDTO
from paralleldomain.common.dgp.v1.point_cloud import PointCloudDTO
from paralleldomain.common.dgp.v1.radar_point_cloud import RadarPointCloudDTO


@dataclass
class SampleCalibrationDTO(DataClassDictMixin):
    names: List[str]
    intrinsics: List[CameraIntrinsicsDTO]
    extrinsics: List[PoseDTO]


@dataclass
class DatumValueDTO(DataClassDictMixin):
    image: ImageDTO = None
    point_cloud: PointCloudDTO = None
    file_datum: FileDatumDTO = None
    radar_point_cloud: RadarPointCloudDTO = None

    class Config(BaseConfig):
        code_generation_options = [TO_DICT_ADD_OMIT_NONE_FLAG]


@dataclass
class DatumDTO(DataClassDictMixin):
    id: DatumIdDTO
    key: str
    datum: DatumValueDTO
    next_key: str
    prev_key: str


@dataclass
class SampleDTO(DataClassDictMixin):
    id: DatumIdDTO
    datum_keys: List[str]
    calibration_key: str
    metadata: Dict[str, AnyDTO]
