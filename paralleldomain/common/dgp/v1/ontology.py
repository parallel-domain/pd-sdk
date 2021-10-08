from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json
from mashumaro import DataClassDictMixin

from paralleldomain.model.class_mapping import ClassMap


@dataclass
class ColorDTO(DataClassDictMixin):
    r: int
    g: int
    b: int


@dataclass
class OntologyItemDTO(DataClassDictMixin):
    name: str
    id: int
    color: ColorDTO
    isthing: bool
    supercategory: str


@dataclass
class OntologyDTO(DataClassDictMixin):
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
