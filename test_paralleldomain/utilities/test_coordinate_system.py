import numpy as np
import pytest

from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.transformation import Transformation


@pytest.mark.parametrize(
    "system,roll,pitch,yaw,order,source,target",
    [
        ("UFL", 0, 90, 90, "xzy", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),  # ryp
        ("UFL", 0, 90, 90, "xyz", np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0])),  # rpy
        ("UFL", 0, 0, 90, "xyz", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),  # rpy
        ("RFU", 0, 0, 90, "xyz", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # rpy
    ],
)
def test_quaternion_transform_90_yaw_and_pitch_xyz(
    system: str, roll: float, pitch: float, yaw: float, order: str, source: np.ndarray, target: np.ndarray
):
    source_coord = CoordinateSystem(system)

    quat = source_coord.quaternion_from_rpy(yaw=yaw, pitch=pitch, roll=roll, degrees=True, order=order)
    rotated = quat.rotate(source)
    assert all(np.isclose(target, rotated))


def test_direction_properties():
    coords = CoordinateSystem("ULB")
    assert np.allclose(coords.forward, np.array([0.0, 0.0, -1.0]))
    assert np.allclose(coords.up, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(coords.left, np.array([0.0, 1.0, 0.0]))

    coords = CoordinateSystem("LBU")
    assert np.allclose(coords.forward, np.array([0.0, -1.0, 0.0]))
    assert np.allclose(coords.up, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(coords.left, np.array([1.0, 0.0, 0.0]))

    coords = CoordinateSystem("BRU")
    assert np.allclose(coords.forward, np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(coords.up, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(coords.left, np.array([0.0, -1.0, 0.0]))

    coords = CoordinateSystem("DFR")
    assert np.allclose(coords.forward, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(coords.up, np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(coords.left, np.array([0.0, 0.0, -1.0]))


class TestChangeTransformationCoordinateSystem:
    def test_it_doesnt_change_an_identity_matrix(self) -> None:
        identity_transformation = Transformation()

        result = CoordinateSystem.change_transformation_coordinate_system(
            transformation=identity_transformation, transformation_system="RFU", target_system="FLU"
        )

        assert np.allclose(result.transformation_matrix, np.identity(4))

    def test_it_changes_translation(self) -> None:
        transformation = Transformation(translation=[1, 2, 3])

        result = CoordinateSystem.change_transformation_coordinate_system(
            transformation=transformation, transformation_system="RFU", target_system="FLU"
        )

        assert np.allclose(result.translation, [2, -1, 3])

    def test_yaw_pitch_roll_flips_signs(self) -> None:
        transformation = Transformation.from_euler_angles(angles=[10, 20, 30], order="yxz", degrees=True)

        result = CoordinateSystem.change_transformation_coordinate_system(
            transformation=transformation, transformation_system="RFU", target_system="FLU"
        )

        assert np.allclose(result.as_euler_angles(order="xyz", degrees=True), [10, -20, 30])

    def test_it_creates_correct_transformation_matrix(self) -> None:
        transformation = Transformation.from_euler_angles(
            angles=[0, 0, 90], order="yxz", degrees=True, translation=[1, 2, 3]
        )
        expected_matrix = [[0, -1, 0, 2], [1, 0, 0, -1], [0, 0, 1, 3], [0, 0, 0, 1]]

        result = CoordinateSystem.change_transformation_coordinate_system(
            transformation=transformation, transformation_system="RFU", target_system="FLU"
        )

        assert np.allclose(result.transformation_matrix, expected_matrix)
