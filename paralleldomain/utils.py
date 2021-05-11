from __future__ import annotations
import numpy as np
from pyquaternion import Quaternion


class Transformation:
    def __init__(self):
        self._matrix = np.eye(4)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, m):
        self._matrix = m

    @property
    def rotation_matrix(self):
        return self.matrix[:3, :3]

    @rotation_matrix.setter
    def rotation_matrix(self, R):
        self.matrix[:3, :3] = R

    @property
    def rotation_quaternion(self):
        return Quaternion(matrix=self.rotation_matrix).elements

    @rotation_quaternion.setter
    def rotation_quaternion(self, q):
        self.rotation_matrix = Quaternion(*q)

    @property
    def translation(self):
        return self.matrix[:3, 3]

    @translation.setter
    def translation(self, t):
        self.matrix[:3, 3] = t

    @staticmethod
    def from_PoseDTO(pose_dto: PoseDTO):
        tf = Transformation()
        tf.rotation_quaternion = [
            pose_dto.rotation.qw,
            pose_dto.rotation.qx,
            pose_dto.rotation.qy,
            pose_dto.rotation.qz,
        ]
        tf.translation = [
            pose_dto.translation.x,
            pose_dto.translation.y,
            pose_dto.translation.z,
        ]

        return tf
