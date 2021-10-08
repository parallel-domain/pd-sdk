from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.geometry import PoseDTO


class ChannelTypeDTO(IntEnum):
    X = 0
    Y = 1
    Z = 2
    V_X = 3
    V_Y = 4
    V_Z = 5
    RCS_DBSM = 6
    EXISTENCE_PROBABILITY = 7
    SENSOR_ID = 8
    COV_XX = 9
    COV_XY = 10
    COV_XZ = 11
    COV_YX = 12
    COV_YY = 13
    COV_YZ = 14
    COV_ZX = 15
    COV_ZY = 16
    COV_ZZ = 17
    RADIAL_DISTANCE = 18
    AZIMUTH_ANGLE = 19
    ELEVATION_ANGLE = 20
    VELOCITY_XS = 21
    RADIAL_DISTANCE_VARIANCE = 22
    AZIMUTH_ANGLE_VARIANCE = 23
    ELEVATION_ANGLE_VARIANCE = 24
    VELOCITY_VARIANCE = 25
    ACCEL_XS = 26
    COUNT_ALIVE = 27
    REFLECTED_POWER_DB = 28
    R = 29
    G = 30
    B = 31
    CLASS_ID = 32
    INSTANCE_ID = 33
    TIMESTAMP = 34


@dataclass
class RadarPointCloudDTO(DataClassDictMixin):
    filename: str
    annotations: Dict[int, str]
    metadata: Dict[str, AnyDTO]
    point_format: List[ChannelTypeDTO]
    point_fields: List[str]
    pose: PoseDTO
