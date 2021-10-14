from dataclasses import dataclass
from typing import Any, Dict

from dataclasses_json import dataclass_json

from paralleldomain.common.dgp.v1.geometry import PoseDTO


@dataclass_json
@dataclass
class ImageDTO:
    filename: str
    height: int
    width: int
    channels: int
    annotations: Dict[int, str]
    metadata: Dict[str, Any]
    pose: PoseDTO
