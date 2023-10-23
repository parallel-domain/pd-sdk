import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def inherit_docs(cls: Optional[type] = None, level: int = 1):
    def decorator(cls):
        cls_base = cls
        for i in range(level):
            cls_base = cls_base.__bases__[0]

        cls.__doc__ = cls_base.__doc__

        return cls

    if isinstance(cls, type):
        return decorator(cls=cls)
    else:
        return decorator


def clip_with_warning(arr: np.ndarray, dtype: type):
    dtype_info = np.iinfo(dtype)
    min_val, max_val = dtype_info.min, dtype_info.max

    if np.any(arr < min_val) or np.any(arr > max_val):
        logger.warning(
            f"Some values are out of bounds for {dtype.__name__}. They will be clipped to range [{min_val}, {max_val}]."
        )

    return np.clip(arr, min_val, max_val)
