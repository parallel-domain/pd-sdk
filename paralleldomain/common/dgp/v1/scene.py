from dataclasses import dataclass
from typing import Any, Dict, List

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.sample import DatumDTO, SampleDTO
from paralleldomain.common.dgp.v1.statistics import DatasetStatisticsDTO
from paralleldomain.common.dgp.v1.timestamp import TimestampDTO


@dataclass_json
@dataclass
class SceneDTO:
    name: str
    description: str
    log: str
    samples: List[SampleDTO]
    metadata: Dict[str, Any]
    data: List[DatumDTO]
    creation_date: TimestampDTO
    ontologies: Dict[int, str]
    statistics: DatasetStatisticsDTO


@dataclass_json
@dataclass
class ScenesDTO:
    scene: List[SceneDTO]


@dataclass_json
@dataclass
class SceneFilesDTO:
    filenames: List[str]
