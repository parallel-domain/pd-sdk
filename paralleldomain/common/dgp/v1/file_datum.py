from dataclasses import dataclass
from typing import Dict

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.any import AnyDTO


@dataclass
class SelfDescribingFileDTO(DataClassDictMixin):
    filename: str
    schema: AnyDTO


@dataclass
class FileDatumDTO(DataClassDictMixin):
    datum: SelfDescribingFileDTO
    annotations: Dict[str, AnyDTO]
