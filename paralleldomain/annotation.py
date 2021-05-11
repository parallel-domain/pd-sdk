from __future__ import annotations as ann
from .utils import Transformation


class AnnotationPose(Transformation):
    @staticmethod
    def from_PoseDTO(pose: PoseDTO):
        tf = Transformation.from_PoseDTO(pose)
        tf.__class__ = AnnotationPose
        return tf


class Annotation:
    ...


class BoundingBox3D(Annotation):
    def __init__(self):
        self._pose: AnnotationPose = AnnotationPose()
        self._width: float = 0.0
        self._length: float = 0.0
        self._height: float = 0.0
        self._class_id: int = None
        self._instance_id: int = None
        self._num_points: int = 0

    def __repr__(self):
        rep = f"Class ID: {self.class_id} {self.pose}"
        return rep

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value: AnnotationPose):
        self._pose = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def heigth(self):
        return self._heigth

    @heigth.setter
    def heigth(self, value):
        self._heigth = value

    @property
    def class_id(self):
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        self._class_id = value

    @property
    def instance_id(self):
        return self._instance_id

    @instance_id.setter
    def instance_id(self, value):
        self._instance_id = value

    @property
    def num_points(self):
        return self._num_points

    @num_points.setter
    def num_points(self, value):
        self._num_points = value

    @staticmethod
    def from_BoundingBox3DDTO(bb3d_dto: BoundingBox3DDTO):
        annotation = BoundingBox3D()
        annotation.pose = AnnotationPose.from_PoseDTO(bb3d_dto.box.pose)
        annotation.width = bb3d_dto.box.width
        annotation.length = bb3d_dto.box.length
        annotation.heigth = bb3d_dto.box.width
        annotation.class_id = bb3d_dto.class_id
        annotation.instance_id = bb3d_dto.instance_id
        annotation.num_points = bb3d_dto.num_points

        return annotation
