import math
import random
from typing import List, Union

import numpy as np
import pytest
from pyquaternion import Quaternion

from paralleldomain.utilities.coordinate_system import CoordinateSystem
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


def test_quaternion_constructor():
    random_state = random.Random(1337)

    quaternion_elements = [
        random_state.uniform(-1, 1),
        random_state.uniform(-1, 1),
        random_state.uniform(-1, 1),
        random_state.uniform(-1, 1),
    ]

    pyquat = Quaternion(
        w=quaternion_elements[0], x=quaternion_elements[1], y=quaternion_elements[2], z=quaternion_elements[3]
    )

    # via List[float]
    tf_0 = Transformation(quaternion=quaternion_elements)
    assert np.allclose(pyquat.elements, tf_0.quaternion.elements)

    # via np.ndarray
    tf_1 = Transformation(quaternion=np.asarray(quaternion_elements))
    assert np.allclose(pyquat.elements, tf_1.quaternion.elements)


_COORDINATE_SYSTEMS = ["FLU", "RUB", "RDF", "RFU"]


class TestLookAt:
    def test_it_can_create_identity(self):
        result = Transformation.look_at(target=[1, 0, 0], position=[0, 0, 0], coordinate_system="FLU")

        assert np.allclose(result.transformation_matrix, np.identity(4))

    @pytest.mark.parametrize("coordinate_system", _COORDINATE_SYSTEMS)
    def test_it_can_look_right(self, coordinate_system: str):
        target = (
            CoordinateSystem.get_base_change_from_to(from_axis_directions="FLU", to_axis_directions=coordinate_system)
            @ np.array([[0, -1, 0]])[0]
        )
        result = Transformation.look_at(target=target, position=[0, 0, 0], coordinate_system=coordinate_system)
        resulting_target = (result @ CoordinateSystem(coordinate_system).forward[np.newaxis, :])[0]

        expected = CoordinateSystem.change_transformation_coordinate_system(
            transformation=Transformation.from_euler_angles(angles=[0, 0, -90], order="yxz", degrees=True),
            transformation_system="FLU",
            target_system=coordinate_system,
        )

        assert np.allclose(resulting_target, target)
        assert np.allclose(
            result.transformation_matrix,
            expected.transformation_matrix,
        )

    @pytest.mark.parametrize("coordinate_system", _COORDINATE_SYSTEMS)
    def test_it_translates_and_rotates(self, coordinate_system: str):
        target = (
            CoordinateSystem.get_base_change_from_to(from_axis_directions="FLU", to_axis_directions=coordinate_system)
            @ np.array([[1 + np.sqrt(3), 2 + np.sqrt(3), 3 + np.sqrt(3)]])
        )[0]
        position = (
            CoordinateSystem.get_base_change_from_to(from_axis_directions="FLU", to_axis_directions=coordinate_system)
            @ np.array([[1, 2, 3]])
        )[0]

        result = Transformation.look_at(target=target, position=position, coordinate_system=coordinate_system)
        direction_to_target = 3 * CoordinateSystem(coordinate_system).forward
        resulting_target = (result @ direction_to_target[np.newaxis, :])[0]

        expected = CoordinateSystem.change_transformation_coordinate_system(
            transformation=Transformation.from_euler_angles(
                angles=[0, -35.26, 45], order="xyz", degrees=True, translation=[1, 2, 3]
            ),
            transformation_system="FLU",
            target_system=coordinate_system,
        )

        assert np.allclose(resulting_target, target)
        assert np.allclose(result.transformation_matrix, expected.transformation_matrix, atol=1e-4)
