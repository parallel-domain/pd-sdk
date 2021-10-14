from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

from dataclasses_json import Undefined, config, dataclass_json

from paralleldomain.common.dgp.v1.scene import SceneFilesDTO
from paralleldomain.common.dgp.v1.statistics import DatasetStatisticsDTO
from paralleldomain.common.dgp.v1.utils import SkipNoneMixin


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
class DatasetMetadataDTO(SkipNoneMixin):
    name: str
    version: str
    creation_date: str
    creator: str
    description: str
    origin: DatasetOriginDTO
    available_annotation_types: List[int]
    frame_per_second: float
    metadata: Any
    statistics: Optional[DatasetStatisticsDTO]
    bucket_path: Optional[str]
    raw_path: Optional[str]


@dataclass_json
@dataclass
class SceneDatasetDTO:
    metadata: DatasetMetadataDTO
    scene_splits: Dict[DatasetSplitDTO, SceneFilesDTO]
