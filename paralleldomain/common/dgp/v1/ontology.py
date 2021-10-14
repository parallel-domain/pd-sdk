from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from paralleldomain.model.class_mapping import ClassMap


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

    @classmethod
    def from_class_map(cls, class_map: ClassMap) -> "OntologyDTO":
        return cls(
            items=[
                OntologyItemDTO(
                    id=cid,
                    name=cval.name,
                    color=ColorDTO(
                        r=cval.meta["color"]["r"],
                        g=cval.meta["color"]["g"],
                        b=cval.meta["color"]["b"],
                    ),
                    isthing=cval.instanced,
                    supercategory="",
                )
                for cid, cval in class_map.items()
            ]
        )
