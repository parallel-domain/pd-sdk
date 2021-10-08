from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.geometry import PoseDTO


class ChannelTypeDTO(Enum):
    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    R = 4
    G = 5
    B = 6
    RING = 7
    NORMAL_X = 8
    NORMAL_Y = 9
    NORMAL_Z = 10
    CLASS_ID = 11
    INSTANCE_ID = 12
    TIMESTAMP = 13

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.name == value:
                return member


@dataclass
class PointCloudDTO(DataClassDictMixin):
    filename: str
    annotations: Dict[int, str]
    metadata: Dict[str, AnyDTO]
    point_format: List[ChannelTypeDTO]
    point_fields: List[str]
    pose: PoseDTO
