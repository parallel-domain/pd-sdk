from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ColorDTO:
    r: int
    g: int
    b: int


@dataclass_json
@dataclass
class OntologyItemDTO:
    name: str
    id: int
    color: ColorDTO
    isthing: bool
    supercategory: str


@dataclass_json
@dataclass
class OntologyDTO:
    items: List[OntologyItemDTO]
