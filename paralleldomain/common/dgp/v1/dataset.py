from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.scene import SceneFilesDTO
from paralleldomain.common.dgp.v1.statistics import DatasetStatisticsDTO
from paralleldomain.common.dgp.v1.timestamp import TimestampDTO


class DatasetSplitDTO(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    TRAIN_OVERFIT = 3


class DatasetOriginDTO(IntEnum):
    PUBLIC = 0
    INTERNAL = 1


@dataclass_json
@dataclass
class DatasetMetadataDTO:
    name: str
    version: str
    creation_date: TimestampDTO
    creator: str
    bucket_path: str
    raw_path: str
    description: str
    origin: DatasetOriginDTO
    available_annotation_types: List[int]
    statistics: DatasetStatisticsDTO
    frame_per_second: float
    metadata: AnyDTO


@dataclass_json
@dataclass
class SceneDatasetDTO:
    metadata: DatasetMetadataDTO
    scene_splits: Dict[DatasetSplitDTO, SceneFilesDTO]
