from typing import Tuple

import numpy as np
import pytest

from paralleldomain.utilities.mask import lookup_values


@pytest.fixture
def integer_grids() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xv, yv = np.meshgrid(np.arange(-10, 10), np.arange(100, 140))
    xv = xv.astype(np.int8)
    yv = yv.astype(np.int8)
    xy = np.stack([xv, yv], axis=-1)

    return xv, yv, xy


def test_lookup_interpolation(integer_grids):
    xv, yv, xy = integer_grids

    indices = np.asarray(
        [
            [0, 0],  # upper-left border
            [xv.shape[1] - 1, xv.shape[0] - 1],  # lower-right border
            [-10, -10],  # out of index bounds
            [(xv.shape[1] - 1) // 2 + 0.5, (xv.shape[1] - 1) // 2 + 0.5],  # float coordinate for bilinear interpolation
        ]
    )

    xv_interp = lookup_values(mask=xv, x=indices[:, 0], y=indices[:, 1], interpolate=True)
    yv_interp = lookup_values(mask=yv, x=indices[:, 0], y=indices[:, 1], interpolate=True)
    xy_interp = lookup_values(mask=xy, x=indices[:, 0], y=indices[:, 1], interpolate=True)

    # Test Output Shapes
    assert xv_interp.shape == (indices.shape[0],)
    assert yv_interp.shape == (indices.shape[0],)
    assert xy_interp.shape == (indices.shape[0], xy.shape[-1])

    # Test Output Values
    assert xv_interp[0] == np.min(xv)  # should be smallest value (at index (0,0))
    assert xv_interp[1] == np.max(xv)  # should be largest value (at index (M,N))
    assert xv_interp[2] == np.min(xv)  # should be smallest value (at index (-10,-10)--[clip]-->(0,0))
    x_3, y_3 = map(int, indices[3])
    assert xv_interp[3] == (
        xv[y_3, x_3] + (xv[y_3, x_3 + 1] - xv[y_3, x_3]) * 0.5
    )  # should be integer value + 1/2 next value in x-direction (constant values along y-axis)
    assert yv_interp[3] == (
        yv[y_3, x_3] + (yv[y_3 + 1, x_3] - yv[y_3, x_3]) * 0.5
    )  # should be integer value + 1/2 next value in y-direction (constant values along x-axis)
