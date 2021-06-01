import math

import numpy as np
import pytest
from paralleldomain.utilities.coordinate_system import CoordinateSystem


@pytest.mark.parametrize("system,roll,pitch,yaw,order,source,target",
                         [("UFL", 0, 90, 90, "ryp", np.array([0., 0., -1.]), np.array([0., 1., 0.])),
                          ("UFL", 0, 90, 90, "rpy", np.array([0., 0., -1.]), np.array([-1., 0., 0.])),
                          ("UFL", 0, 0, 90, "rpy", np.array([0., 0., -1.]), np.array([0., 1., 0.])),
                          ("RFU", 0, 0, 90, "rpy", np.array([1., 0., 0.]), np.array([0., 1., 0.]))
                          ])
def test_quaternion_transform_90_yaw_and_pitch_ryp(system: str, roll: float, pitch: float, yaw: float, order: str,
                                                   source: np.ndarray, target: np.ndarray):
    source_coord = CoordinateSystem(system)

    quat = source_coord.quaternion_from_rpy(yaw=yaw, pitch=pitch, roll=roll, is_degrees=True,
                                            order=order)
    rotated = quat.rotate(source)
    assert all(np.isclose(target, rotated))
