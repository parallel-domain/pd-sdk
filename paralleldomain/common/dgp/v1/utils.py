from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, TypeVar

import ujson
from mashumaro import DataClassDictMixin
from mashumaro.types import GenericSerializableType

KT = TypeVar("KT", int, str)
VT = TypeVar("VT", int, str)


class GenericDict(Dict[KT, VT], GenericSerializableType, DataClassDictMixin):
    def _serialize(self, types) -> Dict[KT, VT]:
        k_type, v_type = types
        if k_type not in (int, str) or v_type not in (int, str):
            raise TypeError
        return {k_type(k): v_type(v) for k, v in self.items()}

    @classmethod
    def _deserialize(cls, value, types) -> "GenericDict[KT, VT]":
        k_type, v_type = types
        if k_type not in (int, str) or v_type not in (int, str):
            raise TypeError
        return cls({k_type(k): v_type(v) for k, v in value.items()})


def _attribute_key_dump(obj: object) -> str:
    return str(obj)


def _attribute_value_dump(obj: object) -> str:
    if isinstance(obj, Dict) or isinstance(obj, List):
        return ujson.dumps(obj, indent=2, escape_forward_slashes=False)
    else:
        return str(obj)
