from __future__ import annotations as ann

from typing import cast

from paralleldomain.dto import PoseDTO, BoundingBox3DDTO

from paralleldomain.utils import Transformation


class AnnotationPose(Transformation):
    @staticmethod
    def from_dto(dto: PoseDTO) -> "AnnotationPose":
        tf = Transformation.from_dto(dto=dto)
        # tf.__class__ = AnnotationPose
        return cast(tf, AnnotationPose)


class Annotation:
    ...


class BoundingBox3D(Annotation):
    def __init__(
        self,
        pose: AnnotationPose,
        width: float,
        height: float,
        length: float,
        class_id: int,
        instance_id: int,
        num_points: int,
    ):
        super().__init__()
        self.pose: AnnotationPose = pose
        self.width: float = width
        self.length: float = length
        self.height: float = height
        self.class_id: int = class_id
        self.instance_id: int = instance_id
        self.num_points: int = num_points

    def __repr__(self):
        rep = f"Class ID: {self.class_id} {self.pose}"
        return rep

    @staticmethod
    def from_dto(dto: BoundingBox3DDTO):
        return BoundingBox3D(
            pose=AnnotationPose.from_dto(dto=dto.box.pose),
            width=dto.box.width,
            length=dto.box.length,
            height=dto.box.width,
            class_id=dto.class_id,
            instance_id=dto.instance_id,
            num_points=dto.num_points,
        )
