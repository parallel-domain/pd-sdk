from typing import Dict, List, Union

import numpy as np


def boolean_mask_by_value(mask: np.ndarray, value: int) -> np.ndarray:
    return boolean_mask_by_values(mask=mask, values=[value])


def boolean_mask_by_values(mask: np.ndarray, values: List[int]) -> np.ndarray:
    return np.isin(mask, values)


def replace_value(mask: np.ndarray, old_value: int, new_value: int) -> np.ndarray:
    return replace_values(mask=mask, value_map={old_value: new_value})


def replace_values(
    mask: np.ndarray, value_map: Dict[int, int], value_min: Union[int, None] = None, value_max: Union[int, None] = None
) -> np.ndarray:
    index_substitutes = np.array(
        [
            value_map.get(item, item)
            for item in range(
                value_min if value_min is not None else np.iinfo(mask.dtype).min,
                (value_max if value_max is not None else np.iinfo(mask.dtype).max) + 1,
            )
        ]
    )

    return index_substitutes[mask]


def encode_int32_as_rgb8(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([mask & 0xFF, mask >> 8 & 0xFF, mask >> 16 & 0xFF], axis=-1).astype(np.uint8)


def encode_2int16_as_rgba8(mask: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [mask[..., [0]] >> 8, mask[..., [0]] & 0xFF, mask[..., [1]] >> 8, mask[..., [1]] & 0xFF], axis=-1
    ).astype(np.uint8)
