from dataclasses import dataclass

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.timestamp import TimestampDTO


@dataclass_json
@dataclass
class DatumIdDTO:
    log: str
    name: str
    timestamp: TimestampDTO
    index: int
