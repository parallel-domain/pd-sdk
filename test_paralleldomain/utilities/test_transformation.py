import math
import random
from typing import List, Union, Tuple

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


def test_transformation_from_axis_angle():
    # run through representative set of axis-angle rotations and compare to quaternion ground truth
    # Test 1
    axis = np.array([1.0, 0.0, 0.0])
    angle = 0.0
    transformation = Transformation.from_axis_angle(axis=axis, angle=angle)
    assert np.allclose(transformation.quaternion.elements, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(transformation.translation == np.array([0.0, 0.0, 0.0]))

    # Test 2
    axis = np.array([1.0, 0.0, 0.0])
    angle = math.pi / 2
    transformation = Transformation.from_axis_angle(axis=axis, angle=angle)
    assert np.allclose(transformation.quaternion.elements, np.array([0.70710678, 0.70710678, 0.0, 0.0]))
    assert np.all(transformation.translation == np.array([0.0, 0.0, 0.0]))

    # Test 3
    axis = np.array([0.0, 1.0, 0.0])
    angle = math.degrees(math.pi / 4)
    translation = np.array([0.1, 0.2, 0.3])
    transformation = Transformation.from_axis_angle(axis=axis, angle=angle, degrees=True, translation=translation)
    assert np.allclose(transformation.quaternion.elements, np.array([0.92387953, 0.0, 0.38268343, 0.0]))
    assert np.all(transformation.translation == np.array([0.1, 0.2, 0.3]))

    # Test 4
    axis = np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
    angle = math.pi / 3
    translation = np.array([1.0, 2.0, 3.0])
    transformation = Transformation.from_axis_angle(axis=axis, angle=angle, translation=translation)
    assert np.allclose(transformation.quaternion.elements, np.array([0.8660254, 0.28867513, 0.28867513, 0.28867513]))
    assert np.all(transformation.translation == np.array([1.0, 2.0, 3.0]))


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


class TestMatmul:
    def test_identity_doesnt_change_transformation(self) -> None:
        identity = Transformation()
        transformation = Transformation.from_euler_angles(
            angles=[0, 0, 90], order="XYZ", translation=[2, 1, 4], degrees=True
        )

        assert np.allclose((identity @ identity).transformation_matrix, identity.transformation_matrix)
        assert np.allclose((transformation @ identity).transformation_matrix, transformation.transformation_matrix)
        assert np.allclose((identity @ transformation).transformation_matrix, transformation.transformation_matrix)

    @pytest.mark.parametrize(
        "angles_a,translation_a,angles_b,translation_b,angles_result,translation_result",
        [
            ([0, 0, 0], [1, 2, 3], [0, 0, 0], [-1, -4, 4], [0, 0, 0], [0, -2, 7]),
            ([0, 0, 90], [1, 2, 3], [0, 0, 0], [-1, -4, 4], [0, 0, 90], [5, 1, 7]),
            ([0, 0, 90], [1, 2, 3], [0, 0, 90], [-1, -4, 4], [0, 0, 180], [5, 1, 7]),
            ([0, 90, 0], [1, 2, 3], [0, 0, 90], [-1, -4, 4], [0, 90, 90], [5, -2, 4]),
        ],
    )
    def test_it_transforms(
        self,
        angles_a: List[float],
        translation_a: List[float],
        angles_b: List[float],
        translation_b: List[float],
        angles_result: List[float],
        translation_result: List[float],
    ) -> None:
        a = Transformation.from_euler_angles(angles=angles_a, order="XYZ", translation=translation_a, degrees=True)
        b = Transformation.from_euler_angles(angles=angles_b, order="XYZ", translation=translation_b, degrees=True)
        expected_result = Transformation.from_euler_angles(
            angles=angles_result, order="XYZ", translation=translation_result, degrees=True
        )

        assert np.allclose((a @ b).transformation_matrix, expected_result.transformation_matrix)


class TestFromYawPitchRoll:
    @pytest.mark.parametrize("axis_directions", ["FLU", "RFU", "RDF", "DFR"])
    @pytest.mark.parametrize("degrees", [True, False])
    def test_zero_angles_produce_identity_matrix(self, axis_directions: str, degrees: bool) -> None:
        coordinate_system = CoordinateSystem(axis_directions=axis_directions)

        transformation = Transformation.from_yaw_pitch_roll(coordinate_system=coordinate_system, degrees=degrees)

        assert np.allclose(transformation.transformation_matrix, np.identity(4))

    @pytest.mark.parametrize(
        "axis_directions,yaw,pitch,roll", [("FLU", 0, 0, 90), ("LFD", 0, 90, 0), ("RDF", 0, 90, 0), ("DFR", 90, 0, 0)]
    )
    def test_rotation_by_90_degree_around_x_works(
        self, axis_directions: str, yaw: float, pitch: float, roll: float
    ) -> None:
        coordinate_system = CoordinateSystem(axis_directions=axis_directions)

        transformation = Transformation.from_yaw_pitch_roll(
            coordinate_system=coordinate_system, yaw=yaw, pitch=pitch, roll=roll, degrees=True
        )

        assert np.allclose(
            transformation.transformation_matrix, [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )

    @pytest.mark.parametrize(
        "axis_directions,yaw,pitch,roll", [("FLU", 0, 90, 0), ("LFD", 0, 0, 90), ("RDF", 90, 0, 0), ("DFR", 0, 0, 90)]
    )
    def test_rotation_by_90_degree_around_y_works(
        self, axis_directions: str, yaw: float, pitch: float, roll: float
    ) -> None:
        coordinate_system = CoordinateSystem(axis_directions=axis_directions)

        transformation = Transformation.from_yaw_pitch_roll(
            coordinate_system=coordinate_system, yaw=yaw, pitch=pitch, roll=roll, degrees=True
        )

        assert np.allclose(
            transformation.transformation_matrix, [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
        )

    @pytest.mark.parametrize(
        "axis_directions,yaw,pitch,roll", [("FLU", 90, 0, 0), ("LFD", 90, 0, 0), ("RDF", 0, 0, 90), ("DFR", 0, 90, 0)]
    )
    def test_rotation_by_90_degree_around_z_works(
        self, axis_directions: str, yaw: float, pitch: float, roll: float
    ) -> None:
        coordinate_system = CoordinateSystem(axis_directions=axis_directions)

        transformation = Transformation.from_yaw_pitch_roll(
            coordinate_system=coordinate_system, yaw=yaw, pitch=pitch, roll=roll, degrees=True
        )

        assert np.allclose(
            transformation.transformation_matrix, [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    @pytest.mark.parametrize(
        "axis_directions,angle_multipliers",
        [("FLU", (1, 1, 1)), ("LFD", (-1, 1, 1)), ("RDF", (-1, -1, 1)), ("RFU", (1, -1, 1)), ("DFR", (-1, -1, 1))],
    )
    @pytest.mark.parametrize(
        "yaw, pitch,roll,expected_flu_quaternion",
        [
            (
                10,
                20,
                30,
                Quaternion(0.9515, 0.2392, 0.1893, 0.0381),
            ),
            (
                190,
                -23,
                3,
                Quaternion(-0.0906, 0.1963, 0.0429, 0.9754),
            ),
        ],
    )
    def test_rotation_order(
        self,
        axis_directions: str,
        yaw: float,
        pitch: float,
        roll: float,
        expected_flu_quaternion: Quaternion,
        angle_multipliers: Tuple[int, int, int],
    ) -> None:
        coordinate_system = CoordinateSystem(axis_directions=axis_directions)

        # The expected quaternion was calculated in flu using the given angles, but changing e.g. between front/back
        # flips the rotation direction of the roll.
        transformation = Transformation.from_yaw_pitch_roll(
            coordinate_system=coordinate_system,
            yaw=yaw * angle_multipliers[0],
            pitch=pitch * angle_multipliers[1],
            roll=roll * angle_multipliers[2],
            degrees=True,
        )

        expected_transformation = CoordinateSystem.change_transformation_coordinate_system(
            transformation=Transformation(quaternion=expected_flu_quaternion),
            transformation_system="FLU",
            target_system=axis_directions,
        )

        assert np.allclose(
            transformation.transformation_matrix, expected_transformation.transformation_matrix, atol=1e-3
        )

    def test_it_sets_the_translation(self) -> None:
        coordinate_system = CoordinateSystem("FLU")
        transformation = Transformation.from_yaw_pitch_roll(coordinate_system=coordinate_system, translation=[1, 2, 3])

        assert np.allclose(transformation.translation, [1, 2, 3])

    @pytest.mark.parametrize("axis_directions", ["FLU", "RFU", "RDF", "DFR"])
    @pytest.mark.parametrize("degrees", [True, False])
    @pytest.mark.parametrize("yaw,pitch,roll", [(90, 0, 0), (0, 90, 0), (0, 0, 90), (10, 20, 30), (-170, 80, 2)])
    def test_converting_back_to_yaw_pitch_roll_returns_initial_input(
        self, axis_directions: str, yaw: float, pitch: float, roll: float, degrees: bool
    ) -> None:
        coordinate_system = CoordinateSystem(axis_directions=axis_directions)
        if degrees is False:
            yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])

        transformation = Transformation.from_yaw_pitch_roll(
            coordinate_system=coordinate_system, yaw=yaw, pitch=pitch, roll=roll, degrees=degrees
        )
        yaw_pitch_roll = transformation.as_yaw_pitch_roll(coordinate_system=coordinate_system, degrees=degrees)

        assert np.allclose(yaw_pitch_roll, [yaw, pitch, roll])
