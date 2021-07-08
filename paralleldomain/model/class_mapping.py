import abc
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, ItemsView, List, Optional, TypeVar, Union

import numpy as np

T = TypeVar("T")
TClassId = TypeVar("TClassId", int, np.ndarray)


class ClassIdMap:
    def __init__(self, class_id_to_class_id: Dict[int, int]):
        self.class_id_to_class_id = class_id_to_class_id

    @property
    def source_ids(self) -> List[int]:
        return list(self.class_id_to_class_id.keys())

    @property
    def target_ids(self) -> List[int]:
        return list(self.class_id_to_class_id.values())

    def items(self) -> ItemsView[int, int]:
        return self.class_id_to_class_id.items()

    def __getitem__(self, key: TClassId) -> TClassId:
        if isinstance(key, int):
            return self.class_id_to_class_id[key]
        else:
            return np.vectorize(self.class_id_to_class_id.get)(key)


@dataclass
class ColorRGB:
    r: int
    g: int
    b: int


@dataclass
class ClassDetail:
    name: str
    id: int
    instanced: bool = False
    color: ColorRGB = field(
        default_factory=lambda: ColorRGB(r=random.randint(0, 255), g=random.randint(0, 255), b=random.randint(0, 255))
    )


class ClassMap:
    def __init__(self, classes: List[ClassDetail]):
        self._class_id_to_class_detail = {c.id: c for c in classes}

    @property
    def class_ids(self) -> List[int]:
        return sorted(list(self._class_id_to_class_detail.keys()))

    @property
    def class_names(self) -> List[str]:
        return [c.name for cid in self.class_ids for c in self._class_id_to_class_detail[cid]]

    def items(self) -> ItemsView[int, ClassDetail]:
        return self._class_id_to_class_detail.items()

    def __getitem__(self, key: int) -> ClassDetail:
        return self._class_id_to_class_detail[key]


class OnLabelNotDefined(Enum):
    RAISE_ERROR = 0
    KEEP_LABEL = 1
    DISCARD_LABEL = 2


class LabelMapping:
    def __init__(self, label_mapping: Dict[str, str], on_not_defined: OnLabelNotDefined):
        self.on_not_defined = on_not_defined
        self._label_mapping = label_mapping

    def items(self) -> ItemsView[str, str]:
        return self._label_mapping.items()

    def __getitem__(self, key: str) -> Optional[str]:
        if key not in self._label_mapping:
            if self.on_not_defined == OnLabelNotDefined.RAISE_ERROR:
                raise KeyError(f"Missing Mapping for {key}!")
            elif self.on_not_defined == OnLabelNotDefined.KEEP_LABEL:
                return key
            else:
                return None
        return self._label_mapping[key]

    def __matmul__(self, other: T) -> T:
        if isinstance(other, ClassMap):
            return ClassMap(
                classes=[
                    ClassDetail(
                        id=class_id,
                        name=self[class_detail.name],
                        instanced=class_detail.instanced,
                        color=class_detail.color,
                    )
                    for class_id, class_detail in other.items()
                    if self[class_detail.name] is not None
                ]
            )
        elif isinstance(other, LabelMapping):
            return LabelMapping(
                label_mapping={
                    class_name: self[to_class_name]
                    for class_name, to_class_name in other.items()
                    if self[to_class_name] is not None
                },
                on_not_defined=self.on_not_defined,
            )
        else:
            raise ValueError(f"Unsupported type {type(other)}")
