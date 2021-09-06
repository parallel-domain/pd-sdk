import logging
import math
from typing import Dict

import numpy as np
from pyquaternion import Quaternion

logger = logging.getLogger(__name__)


class CoordinateSystem:
    _axis_char_map: Dict[str, np.ndarray] = dict(
        **{character: axis for character, axis in zip("FLU", np.identity(3))},
        **{character: axis for character, axis in zip("BRD", -np.identity(3))},
    )

    def __init__(self, axis_directions: str):
        self.axis_directions = axis_directions
        self._base_matrix = self._create_base_matrix(axis_directions=axis_directions)

    @staticmethod
    def _create_base_matrix(axis_directions: str) -> np.ndarray:
        if len(axis_directions) != 3:
            raise ValueError("The axis string needs to have exactly 3 orthogonal values from {RUBLDF}!")
        axis_directions = axis_directions.upper()
        base_change = np.identity(4)
        for i, direction in enumerate(axis_directions):
            base_change[:3, i] = CoordinateSystem._axis_char_map[direction]
        if np.linalg.det(base_change) == 0.0:
            raise ValueError(f"{axis_directions} is not a valid coordinate system!")
        return base_change

    def __gt__(self, other: "CoordinateSystem") -> np.ndarray:
        return other._base_matrix.transpose() @ self._base_matrix

    def __lt__(self, other: "CoordinateSystem") -> np.ndarray:
        return other > self

    @staticmethod
    def get_base_change_from_to(from_axis_directions: str, to_axis_directions: str) -> np.ndarray:
        return CoordinateSystem(from_axis_directions) > CoordinateSystem(to_axis_directions)

    @staticmethod
    def print_convention():
        logger.info(f"Front axis: {CoordinateSystem._axis_char_map['F']}")
        logger.info(f"Left axis: {CoordinateSystem._axis_char_map['L']}")
        logger.info(f"Up axis: {CoordinateSystem._axis_char_map['U']}")

    def quaternion_from_rpy(self, roll: float, pitch: float, yaw: float, degrees: bool = False, order: str = "xyz"):
        transform = CoordinateSystem("FLU") > self

        front = transform[:3, :3] @ CoordinateSystem._axis_char_map["F"].reshape((3, 1))
        left = transform[:3, :3] @ CoordinateSystem._axis_char_map["L"].reshape((3, 1))
        up = transform[:3, :3] @ CoordinateSystem._axis_char_map["U"].reshape((3, 1))

        rotations = {
            "x": Quaternion(axis=front, radians=roll if not degrees else math.radians(roll)),
            "y": Quaternion(axis=left, radians=pitch if not degrees else math.radians(pitch)),
            "z": Quaternion(axis=up, radians=yaw if not degrees else math.radians(yaw)),
        }
        q = Quaternion()
        for rot in order:
            q = q * rotations[rot]
        return q


INTERNAL_COORDINATE_SYSTEM = CoordinateSystem("FLU")
