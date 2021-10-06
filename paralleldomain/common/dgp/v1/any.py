from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AnyDTO:
    type_url: str
    value: bytes
