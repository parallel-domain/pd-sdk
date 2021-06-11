import abc
from enum import Enum
from typing import Dict, Union, TypeVar, List, ItemsView, Optional

T = TypeVar("T")


class ClassMap:
    def __init__(self, class_id_to_class_name: Dict[int, str]):
        self._class_id_to_class_name = class_id_to_class_name

    @property
    def class_ids(self) -> List[int]:
        return list(self._class_id_to_class_name.keys())

    @property
    def class_names(self) -> List[str]:
        return list(self._class_id_to_class_name.values())

    def items(self) -> ItemsView[int, str]:
        return self._class_id_to_class_name.items()

    def __getitem__(self, key: int) -> str:
        return self._class_id_to_class_name[key]


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
            return ClassMap(class_id_to_class_name={
                class_id: self[class_name] for class_id, class_name in other.items() if self[class_name] is not None
            })
        elif isinstance(other, LabelMapping):
            return LabelMapping(label_mapping={
                class_name: self[to_class_name] for class_name, to_class_name in other.items() if
                self[to_class_name] is not None
            }, on_not_defined=self.on_not_defined)
        else:
            raise ValueError(f"Unsupported type {type(other)}")
