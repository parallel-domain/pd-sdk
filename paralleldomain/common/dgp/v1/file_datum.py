from dataclasses import dataclass
from typing import Any, Dict

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SelfDescribingFileDTO:
    filename: str
    schema: Any


@dataclass_json
@dataclass
class FileDatumDTO:
    datum: SelfDescribingFileDTO
    annotations: Dict[str, Any]
