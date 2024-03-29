from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, ItemsView, List, Optional, TypeVar, Union

import numpy as np

T = TypeVar("T", bound=Union["ClassMap", "LabelMapping"])
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

    def __matmul__(self, other: TClassId) -> TClassId:
        return self[other]


@dataclass
class ClassDetail:
    name: str
    id: int
    instanced: Optional[bool] = False  # TODO deprecate this parameter
    meta: Dict[str, Any] = field(default_factory=lambda: {})


class ClassMap:
    def __init__(self, classes: List[ClassDetail]):
        self._class_id_to_class_detail = {c.id: c for c in classes}

    @property
    def class_details(self) -> List[ClassDetail]:
        return list(self._class_id_to_class_detail.values())

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

    def __len__(self):
        return len(self._class_id_to_class_detail)

    def get_class_detail_from_name(self, class_name: str) -> Optional[ClassDetail]:
        return next(
            iter(
                [
                    self._class_id_to_class_detail[cid]
                    for cid in self.class_ids
                    if self._class_id_to_class_detail[cid].name == class_name
                ]
            ),
            None,
        )

    @staticmethod
    def from_id_label_dict(id_label_dict: Dict[int, str]) -> "ClassMap":
        return ClassMap(classes=[ClassDetail(id=k, name=v) for k, v in id_label_dict.items()])


class OnLabelNotDefined(Enum):
    RAISE_ERROR = 0
    KEEP_LABEL = 1
    DISCARD_LABEL = 2
    MAP_TO_DEFAULT = 3


class LabelMapping:
    def __init__(self, label_mapping: Dict[str, str], on_not_defined: OnLabelNotDefined, default_name: str = None):
        self.on_not_defined = on_not_defined
        self._label_mapping = label_mapping
        self.default_name = default_name
        if on_not_defined == OnLabelNotDefined.MAP_TO_DEFAULT and default_name is None:
            raise ValueError("When using MAP_TO_DEFAULT you also need to provide a default name!")

    def items(self) -> ItemsView[str, str]:
        return self._label_mapping.items()

    def __getitem__(self, key: str) -> Optional[str]:
        if key not in self._label_mapping:
            if self.on_not_defined == OnLabelNotDefined.RAISE_ERROR:
                raise KeyError(f"Missing Mapping for {key}!")
            elif self.on_not_defined == OnLabelNotDefined.KEEP_LABEL:
                return key
            elif self.on_not_defined == OnLabelNotDefined.MAP_TO_DEFAULT:
                return self.default_name
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


TClassNameToIdMappable = TypeVar("TClassNameToIdMappable", ClassMap, LabelMapping)


class ClassNameToIdMap:
    def __init__(self, name_to_class_id: Dict[str, int], on_not_defined: OnLabelNotDefined, default_id: int = None):
        self.default_id = default_id
        self.on_not_defined = on_not_defined
        self.name_to_class_id = name_to_class_id
        assert on_not_defined != OnLabelNotDefined.KEEP_LABEL, "KEEP_LABEL is not supported!"

    def __getitem__(self, key: str) -> Optional[int]:
        if key not in self.name_to_class_id:
            if self.on_not_defined == OnLabelNotDefined.RAISE_ERROR:
                raise KeyError(f"Missing Mapping for {key}!")
            elif self.on_not_defined == OnLabelNotDefined.MAP_TO_DEFAULT:
                return self.default_id
            else:
                return None
        return self.name_to_class_id[key]

    def __matmul__(self, other: TClassNameToIdMappable) -> Union[ClassIdMap, "ClassNameToIdMap"]:
        if isinstance(other, ClassMap):
            mapping: Dict[int, int] = dict()
            for source_id, class_detail in other.items():
                mapped = self[class_detail.name]
                if mapped is not None:
                    mapping[source_id] = mapped
            return ClassIdMap(class_id_to_class_id=mapping)
        if isinstance(other, LabelMapping):
            adjusted_name_to_class_id: Dict[str, int] = dict()
            for class_name, to_class_name in other.items():
                mapped = self[to_class_name]
                if mapped is not None:
                    adjusted_name_to_class_id[class_name] = mapped
            return ClassNameToIdMap(
                name_to_class_id=adjusted_name_to_class_id,
                on_not_defined=self.on_not_defined,
                default_id=self.default_id,
            )
