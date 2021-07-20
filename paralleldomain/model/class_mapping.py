from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, ItemsView, List, Optional, TypeVar

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
class ClassDetail:
    name: str
    id: int
    instanced: bool = False
    meta: Dict[str, Any] = field(default_factory=lambda: {})


class ClassMap:
    def __init__(self, classes: List[ClassDetail]):
        self._class_id_to_class_detail = {c.id: c for c in classes}

    @property
    def class_ids(self) -> List[int]:
        return sorted(list(self._class_id_to_class_detail.keys()))

    @property
    def class_names(self) -> List[str]:
        return [self._class_id_to_class_detail[cid].name for cid in self.class_ids]

    def items(self) -> ItemsView[int, ClassDetail]:
        return self._class_id_to_class_detail.items()

    def __getitem__(self, key: int) -> ClassDetail:
        return self._class_id_to_class_detail[key]

    @staticmethod
    def from_id_label_dict(id_label_dict: Dict[int, str]) -> "ClassMap":
        return ClassMap(classes=[ClassDetail(id=k, name=v) for k, v in id_label_dict.items()])


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
                        meta=class_detail.meta,
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
