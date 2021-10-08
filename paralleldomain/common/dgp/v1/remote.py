from dataclasses import dataclass

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin


@dataclass
class RemotePathDTO(DataClassDictMixin):
    value: str


@dataclass
class RemoteArtifactDTO(DataClassDictMixin):
    url: RemotePathDTO
    sha1: str
    isdir: bool
