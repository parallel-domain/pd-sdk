import numpy as np
import pytest

from paralleldomain.utilities.coordinate_system import CoordinateSystem


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
