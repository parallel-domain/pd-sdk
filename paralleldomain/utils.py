from __future__ import annotations as ann
import numpy as np
from pyquaternion import Quaternion


class Transformation:
    def __init__(self):
        self._Rq = Quaternion(1, 0, 0, 0)
        self._t = [0.0, 0.0, 0.0]

    def __repr__(self):
        rep = f"R: {self.rpy}, t: {self.translation}"
        return rep

    def __matmul__(self, other):
        if isinstance(other, Transformation):
            return np.dot(self.matrix, other.matrix)
        elif isinstance(other, np.ndarray):
            return np.dot(self.matrix, other)

    @property
    def transformation_matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    @transformation_matrix.setter
    def transformation_matrix(self, m):
        self._matrix = m

    @property
    def rotation(self):
        return self._Rq.rotation_matrix

    @rotation.setter
    def rotation(self, R):
        self._Rq = Quaternion(matrix=R)

    @property
    def rotation_quaternion(self):
        return self._Rq.elements

    @rotation_quaternion.setter
    def rotation_quaternion(self, q):
        self._Rq = Quaternion(*q)

    @property
    def rpy(self):
        return [
            self._Rq.yaw_pitch_roll[2],
            self._Rq.yaw_pitch_roll[1],
            self._Rq.yaw_pitch_roll[0],
        ]

    @property
    def translation(self):
        return self._t

    @translation.setter
    def translation(self, t):
        self._t = t

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
