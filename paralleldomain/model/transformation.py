from __future__ import annotations as ann

from typing import List

import numpy as np
from pyquaternion import Quaternion


class Transformation:
    def __init__(self, quaternion: Quaternion, translation: np.ndarray):
        self._Rq = quaternion
        self._t = translation

    def __repr__(self):
        rep = f"R: {self.rpy}, t: {self.translation}"
        return rep

    def __matmul__(self, other) -> "Transformation":
        if isinstance(other, Transformation):
            transform = self.transformation_matrix @ other.transformation_matrix
        elif isinstance(other, np.ndarray):
            transform = self.transformation_matrix @ other
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4x4 numpy array!")
        return Transformation.from_transformation_matrix(mat=transform)

    def __rmatmul__(self, other) -> "Transformation":
        if isinstance(other, Transformation):
            transform = other.transformation_matrix @ self.transformation_matrix
        elif isinstance(other, np.ndarray):
            transform = other @ self.transformation_matrix
        else:
            raise ValueError(f"Invalid value {other}! Has to be a Transformation or 4x4 numpy array!")
        return Transformation.from_transformation_matrix(mat=transform)

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

    @staticmethod
    def from_transformation_matrix(mat: np.ndarray) -> "Transformation":
        quat = Quaternion(matrix=mat)
        translation = mat[:3, 3]
        return Transformation(quaternion=quat, translation=translation)