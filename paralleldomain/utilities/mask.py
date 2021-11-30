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


def encode_rgb8_as_int32(mask: np.ndarray) -> np.ndarray:
    return (mask[..., 2:3] << 16) + (mask[..., 1:2] << 8) + mask[..., 0:1]


def encode_2int16_as_rgba8(mask: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [mask[..., [0]] >> 8, mask[..., [0]] & 0xFF, mask[..., [1]] >> 8, mask[..., [1]] & 0xFF], axis=-1
    ).astype(np.uint8)


def lookup_values(
    mask: np.ndarray, x: Union[np.ndarray, List], y: Union[np.ndarray, List], interpolate: bool = False
) -> np.ndarray:
    """Executes bilinear interpolation on a 2D plane.

    Args:
        mask: Array of shape (M x N [x L]). Note: If 3 dimensions are provided,
            bilinear interpolation is performed on each 2D plane in the first two dimensions.
        x: List of indices to interpolate on along the x-axis (columns).
            Indices < 0 and > (M-1, N-1) will be clipped to 0 or (M-1, N-1), respectively.
        y: List of indices to interpolate on along the y-axis (rows).
            Indices < 0 and > (M-1, N-1) will be clipped to 0 or (M-1, N-1), respectively.

    Returns:
        Returns interpolated values for input (x,y) as array with shape (len(x) [x L]).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim > 1 and x.shape[1] > 1:
        raise ValueError(f"Expecting shapes (N) or (N x 1) for `x`, received {x.shape}.")
    if y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(f"Expecting shapes (N) or (N x 1) for `y`, received {y.shape}.")
    if x.shape != y.shape:
        raise ValueError(f"Both `x` and `y` must have same shapes, received x: {x.shape} and y: {y.shape}.")

    x = x.reshape(-1)
    y = y.reshape(-1)

    if interpolate:
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, mask.shape[1] - 1)
        x1 = np.clip(x1, 0, mask.shape[1] - 1)
        y0 = np.clip(y0, 0, mask.shape[0] - 1)
        y1 = np.clip(y1, 0, mask.shape[0] - 1)

        Ia = mask[y0, x0]
        Ib = mask[y1, x0]
        Ic = mask[y0, x1]
        Id = mask[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        interpolated_result = (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T
        border_cases = np.logical_or(x0 == x1, y0 == y1)
        interpolated_result[border_cases] = mask[y0[border_cases], x0[border_cases]]

        return interpolated_result
    else:
        return mask[y.astype(int), x.astype(int)]
