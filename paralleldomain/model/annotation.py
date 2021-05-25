from __future__ import annotations as ann

from typing import Type

from paralleldomain.model.transformation import Transformation


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


class BoundingBox2D(Annotation):
    ...


class SemanticSegmentation2D(Annotation):
    ...


class BoundingBox3D(Annotation):
    def __init__(
        self,
        pose: AnnotationPose,
        width: float,
        height: float,
        length: float,
        class_id: int,
        class_name: str,
        instance_id: int,
        num_points: int,
    ):
        super().__init__()
        self.pose: AnnotationPose = pose
        self.width: float = width
        self.length: float = length
        self.height: float = height
        self.class_id: int = class_id
        self.class_name: str = class_name
        self.instance_id: int = instance_id
        self.num_points: int = num_points

    def __repr__(self):
        rep = f"Class ID: {self.class_id} {self.pose}"
        return rep


AnnotationType = Type[Annotation]


class AnnotationTypes:
    BoundingBox2D: Type[BoundingBox2D] = BoundingBox2D
    BoundingBox3D: Type[BoundingBox3D] = BoundingBox3D
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D
