from dataclasses import dataclass

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin


@dataclass
class AnyDTO(DataClassDictMixin):
    type_url: str
    value: bytes
