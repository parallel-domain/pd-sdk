from dataclasses import dataclass
from typing import Dict, List

from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.sample import DatumDTO, SampleDTO
from paralleldomain.common.dgp.v1.statistics import DatasetStatisticsDTO
from paralleldomain.common.dgp.v1.timestamp import TimestampDTO
from paralleldomain.common.dgp.v1.utils import GenericDict


@dataclass
class SceneDTO(DataClassDictMixin):
    name: str
    description: str
    log: str
    samples: List[SampleDTO]
    metadata: Dict[str, AnyDTO]
    data: List[DatumDTO]
    creation_date: TimestampDTO
    ontologies: Dict[int, str]
    statistics: DatasetStatisticsDTO


@dataclass
class ScenesDTO(DataClassDictMixin):
    scene: List[SceneDTO]


@dataclass
class SceneFilesDTO(DataClassDictMixin):
    filenames: List[str]
