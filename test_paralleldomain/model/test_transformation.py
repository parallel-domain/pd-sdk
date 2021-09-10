import math
import random
from typing import List, Union

import numpy as np
import pytest
from pyquaternion import Quaternion

from paralleldomain.utilities.transformation import Transformation


# Ground Truth Values pre-calculated using scipy.spatial.transform.rotation.Rotation class
@pytest.mark.parametrize(
    "angles,order,translation,degrees,target",
    [
        (
            [0, 90, -90],
            "xyz",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([0.5000000000000001, 0.4999999999999999, 0.5, -0.5]),
        ),
        (
            [180, 90, -90],
            "zyx",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([0.49999999999999994, 0.49999999999999994, 0.5, 0.5000000000000001]),
        ),
        (
            [0, -90, 90],
            "xzy",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([0.5000000000000001, -0.4999999999999999, 0.5, -0.5]),
        ),
        (
            [90, 90, 0],
            "XYZ",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([0.5000000000000001, 0.5, 0.5, 0.4999999999999999]),
        ),
        (
            [180, 90, -180],
            "ZYX",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([-0.7071067811865475, -8.659560562354933e-17, -0.7071067811865476, 8.659560562354933e-17]),
        ),
        (
            [0, -90, 180],
            "XZY",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([4.329780281177467e-17, 0.7071067811865475, 0.7071067811865476, -4.329780281177466e-17]),
        ),
        (
            [90, 90, -180],
            "zxz",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([0.5, -0.49999999999999983, -0.5, -0.5000000000000001]),
        ),
        (
            [180, 0, 0],
            "XZX",
            np.array([0.0, 0.0, 0.0]),
            True,
            Quaternion([6.123233995736766e-17, 1.0, 0.0, 0.0]),
        ),
    ],
)
def test_transformation_from_rpy(
    angles: Union[np.ndarray, List[float]], order: str, translation: np.ndarray, degrees: bool, target: Quaternion
):
    transform = Transformation.from_euler_angles(angles=angles, translation=translation, order=order, degrees=True)
    assert np.all(np.isclose(target.rotation_matrix, transform.rotation))


def test_transformation_inverse():
    random_state = random.Random(1337)

    # Identity Matrix (no rotation, no translation)
    trans_0 = Transformation.from_transformation_matrix(np.eye(4))
    trans_0_inverse = trans_0.inverse
    trans_0_identity = trans_0_inverse @ trans_0
    assert np.allclose(trans_0_identity.transformation_matrix, np.eye(4))

    # Random Rotation (no translation)
    trans_1 = Transformation.from_euler_angles(
        angles=[
            random_state.uniform(-1, 1) * math.pi,
            random_state.uniform(-1, 1) * math.pi,
            random_state.uniform(-1, 1) * math.pi,
        ],
        order="xyz",
        degrees=False,
    )
    trans_1_inverse = trans_1.inverse
    trans_1_identity = trans_1_inverse @ trans_1
    assert np.allclose(trans_1_identity.transformation_matrix, np.eye(4))

    # Random Transformation
    trans_2 = Transformation.from_euler_angles(
        angles=[
            random_state.uniform(-1, 1) * math.pi,
            random_state.uniform(-1, 1) * math.pi,
            random_state.uniform(-1, 1) * math.pi,
        ],
        translation=np.asarray(
            [random_state.uniform(-200, 200), random_state.uniform(-200, 200), random_state.uniform(-200, 200)]
        ),
        order="ZYX",
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
