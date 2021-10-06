from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class RemotePathDTO:
    value: str


@dataclass_json
@dataclass
class RemoteArtifactDTO:
    url: RemotePathDTO
    sha1: str
    isdir: bool
