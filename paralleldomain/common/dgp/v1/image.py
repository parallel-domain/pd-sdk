from dataclasses import dataclass
from typing import Dict

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin

from paralleldomain.common.dgp.v1.any import AnyDTO
from paralleldomain.common.dgp.v1.geometry import PoseDTO


@dataclass
class ImageDTO(DataClassDictMixin):
    filename: str
    height: int
    width: int
    channels: int
    annotations: Dict[int, str]
    metadata: Dict[str, AnyDTO]
    pose: PoseDTO
