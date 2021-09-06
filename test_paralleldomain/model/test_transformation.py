import math
import random
from typing import Optional, Union

import numpy as np
import pytest
from pyquaternion import Quaternion

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
            "xzy",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            "UFL",
            0,
            90,
            90,
            "xyz",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 1.0]),
            np.array([-1.0, 0.0, 0.0, 1.0]),
        ),
        (
            "UFL",
            0,
            0,
            90,
            "xyz",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            "RFU",
            0,
            0,
            90,
            "xyz",
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            None,
            0,
            0,
            90,
            "xyz",
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        ),
        (
            None,
            0,
            0,
            90,
            "xyz",
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
        roll=roll, pitch=pitch, yaw=yaw, translation=translation, degrees=True, order=order, coordinate_system=system
    )
    transformed = transform @ source
    assert all(np.isclose(target, transformed))


def test_transformation_inverse():
    random_state = random.Random(1337)

    # Identity Matrix (no rotation, no translation)
    trans_0 = Transformation.from_transformation_matrix(np.eye(4))
    trans_0_inverse = trans_0.inverse
    trans_0_identity = trans_0_inverse @ trans_0
    assert np.allclose(trans_0_identity.transformation_matrix, np.eye(4))

    # Random Rotation (no translation)
    trans_1 = Transformation.from_euler_angles(
        roll=random_state.uniform(-1, 1) * math.pi,
        pitch=random_state.uniform(-1, 1) * math.pi,
        yaw=random_state.uniform(-1, 1) * math.pi,
        degrees=False,
    )
    trans_1_inverse = trans_1.inverse
    trans_1_identity = trans_1_inverse @ trans_1
    assert np.allclose(trans_1_identity.transformation_matrix, np.eye(4))

    # Random Transformation
    trans_2 = Transformation.from_euler_angles(
        roll=random_state.uniform(-1, 1) * math.pi,
        pitch=random_state.uniform(-1, 1) * math.pi,
        yaw=random_state.uniform(-1, 1) * math.pi,
        translation=np.asarray(
            [random_state.uniform(-200, 200), random_state.uniform(-200, 200), random_state.uniform(-200, 200)]
        ),
        degrees=False,
    )
    trans_2_inverse = trans_2.inverse
    trans_2_identity = trans_2_inverse @ trans_2
    assert np.allclose(trans_2_identity.transformation_matrix, np.eye(4))

    # Random Translation (no rotation)
    trans_3 = Transformation(
        Quaternion(),
        np.asarray([random_state.uniform(-200, 200), random_state.uniform(-200, 200), random_state.uniform(-200, 200)]),
    )
    trans_3_inverse = trans_3.inverse
    trans_3_identity = trans_3_inverse @ trans_3
    assert np.allclose(trans_3_identity.transformation_matrix, np.eye(4))
