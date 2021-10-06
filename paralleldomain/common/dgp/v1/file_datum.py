from dataclasses import dataclass
from typing import Dict

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.any import AnyDTO


@dataclass_json
@dataclass
class SelfDescribingFileDTO:
    filename: str
    schema: AnyDTO


@dataclass_json
@dataclass
class FileDatumDTO:
    datum: SelfDescribingFileDTO
    annotations: Dict[str, AnyDTO]
