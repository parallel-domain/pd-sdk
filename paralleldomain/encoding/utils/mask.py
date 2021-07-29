from typing import List

import numpy as np


def boolean_mask_by_value(mask: np.ndarray, value: object) -> np.ndarray:
    return boolean_mask_by_values(mask=mask, values=[value])


def boolean_mask_by_values(mask: np.ndarray, values: List[object]) -> np.ndarray:
    return np.isin(mask, values)


def replace_value(mask: np.ndarray, old_value: object, new_value: object) -> np.ndarray:
    return replace_values(mask=mask, old_values=[old_value], new_value=new_value)


def replace_values(mask: np.ndarray, old_values: List[object], new_value: object) -> np.ndarray:
    boolean_mask = boolean_mask_by_values(mask=mask, values=old_values)
    mask[boolean_mask] = new_value
    return mask


def encode_as_rgb8(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([mask & 0xFF, mask >> 8 & 0xFF, mask >> 16 & 0xFF], axis=-1).astype(np.uint8)
