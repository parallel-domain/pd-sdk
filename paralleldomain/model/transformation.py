from __future__ import annotations as ann

from typing import List, Optional, Union

import numpy as np
from pyquaternion import Quaternion

from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, CoordinateSystem


class Transformation:
    def __init__(self, quaternion: Quaternion = None, translation: Union[np.ndarray, List] = None):
        self._Rq = quaternion if quaternion is not None else Quaternion(w=1, x=0, y=0, z=0)
        self._t = (
            np.asarray(translation).reshape(
                3,
            )
            if translation is not None
            else np.array([0, 0, 0])
        )

    def __repr__(self):
        rep = f"R: {self.rpy}, t: {self.translation}"
        return rep

    def __matmul__(self, other) -> Union["Transformation", np.ndarray]:
        convert_to_transform = True
        if isinstance(other, Transformation):
            transform = self.transformation_matrix @ other.transformation_matrix
        elif isinstance(other, np.ndarray):
            transform = self.transformation_matrix @ other
            convert_to_transform = other.shape == (4, 4)
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4x4 numpy array!")
        if convert_to_transform:
            return Transformation.from_transformation_matrix(mat=transform)
        return transform

    def __rmatmul__(self, other) -> Union["Transformation", np.ndarray]:
        convert_to_transform = True
        if isinstance(other, Transformation):
            transform = other.transformation_matrix @ self.transformation_matrix
        elif isinstance(other, np.ndarray):
            transform = other @ self.transformation_matrix
            convert_to_transform = other.shape == (4, 4)
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4x4 numpy array!")

        if convert_to_transform:
            return Transformation.from_transformation_matrix(mat=transform)
        return transform

    @property
    def transformation_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    @property
    def rotation(self) -> np.ndarray:
        return self._Rq.rotation_matrix

    @property
    def quaternion(self) -> Quaternion:
        return self._Rq

    @property
    def rotation_quaternion(self) -> np.ndarray:
        return self._Rq.elements

    @property
    def rpy(self) -> List[float]:
        return [
            self._Rq.yaw_pitch_roll[2],
            self._Rq.yaw_pitch_roll[1],
            self._Rq.yaw_pitch_roll[0],
        ]

    @property
    def translation(self) -> np.ndarray:
        return self._t

    @property
    def inverse(self) -> "Transformation":
        q_inv = self.quaternion.inverse
        t_inv = -1 * q_inv.rotate(self.translation)
        T_inv = Transformation(q_inv, t_inv)
        return T_inv

    @classmethod
    def from_transformation_matrix(cls, mat: np.ndarray) -> "Transformation":
        quat = Quaternion(matrix=mat)
        translation = mat[:3, 3]
        return cls(quaternion=quat, translation=translation)

    @classmethod
    def from_euler_angles(
        cls,
        yaw: float,
        pitch: float,
        roll: float,
        translation: Optional[np.ndarray] = None,
        is_degrees: bool = False,
        order: str = "rpy",
        coordinate_system: Optional[Union[str, CoordinateSystem]] = None,
    ) -> "Transformation":
        if translation is None:
            translation = np.array([0.0, 0.0, 0.0])

        if coordinate_system is None:
            coordinate_system = INTERNAL_COORDINATE_SYSTEM
        elif isinstance(coordinate_system, str):
            coordinate_system = CoordinateSystem(axis_directions=coordinate_system)

        quat = coordinate_system.quaternion_from_rpy(
            yaw=yaw, pitch=pitch, roll=roll, is_degrees=is_degrees, order=order
        )
        return cls(quaternion=quat, translation=translation)
