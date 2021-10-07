import dataclasses
from typing import Dict

import numpy as np


def deserialize(v):
    return [np.uint32(vv) for vv in v]


NP_UINT32: Dict = dict(deserialize=np.uint32, serialize=int)
LIST_NP_UINT32: Dict = dict(deserialize=lambda v: list(map(np.uint32, v)), serialize=lambda v: list(map(int, v)))

DICT_STR_STR: Dict = dict(
    deserialize=lambda kv: {str(k): str(v) for k, v in kv.items()},
    serialize=lambda kv: {str(k): str(v) for k, v in kv.items()},
)
