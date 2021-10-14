from dataclasses import dataclass
from datetime import datetime
from math import modf

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TimestampDTO:
    seconds: int
    nanos: int

    def __lt__(self, other):
        if isinstance(other, TimestampDTO):
            return (self.seconds + (self.nanos * 10 ** -9)) < (other.seconds + (other.nanos * 10 ** -9))
        else:
            return self < other

    def __gt__(self, other):
        if isinstance(other, TimestampDTO):
            return (self.seconds + (self.nanos * 10 ** -9)) > (other.seconds + (other.nanos * 10 ** -9))
        else:
            return self > other

    @classmethod
    def from_datetime(cls, dt: datetime):
        seconds, microseconds = map(int, modf(dt.timestamp()))
        return cls(seconds=seconds, nanos=(microseconds * 1000))
