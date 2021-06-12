from typing import Optional, Union

import numpy as np
import pytest

from paralleldomain.model.transformation import Transformation
from paralleldomain.utilities.coordinate_system import CoordinateSystem


@pytest.mark.parametrize(
    "system,roll,pitch,yaw,order,translation,source,target",
    [
        (
            "UFL",
            0,
            90,
            90,
            "ryp",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            "UFL",
            0,
            90,
            90,
            "rpy",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 1.0]),
            np.array([-1.0, 0.0, 0.0, 1.0]),
        ),
        (
            "UFL",
            0,
            0,
            90,
            "rpy",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            "RFU",
            0,
            0,
            90,
            "rpy",
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            None,
            0,
            0,
            90,
            "rpy",
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            None,
            0,
            0,
            90,
            "rpy",
            np.array([42.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([42.0, 1.0, 0.0, 1.0]),
        ),
    ],
)
def test_transformation_from_rpy(
    system: Optional[Union[str, CoordinateSystem]],
    roll: float,
    pitch: float,
    yaw: float,
    order: str,
    translation: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
):
    transform = Transformation.from_euler_angles(
        yaw=yaw, pitch=pitch, roll=roll, translation=translation, is_degrees=True, order=order, coordinate_system=system
    )
    transformed = transform @ source
    assert all(np.isclose(target, transformed))
