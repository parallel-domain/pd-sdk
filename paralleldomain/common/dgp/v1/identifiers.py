from dataclasses import dataclass

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.timestamp import TimestampDTO


@dataclass
class DatumIdDTO(DataClassDictMixin):
    log: str
    name: str
    timestamp: TimestampDTO
    index: int
