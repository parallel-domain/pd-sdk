from __future__ import annotations as ann

import numpy as np
from dataclasses import dataclass, field
from typing import Type, List, Any, Dict

from paralleldomain.model.transformation import Transformation

import numpy as np


class AnnotationPose(Transformation):
    ...


class Annotation:
    ...


class BoundingBox2D(Annotation):
    ...


@dataclass
class SemanticSegmentation2D(Annotation):
    mask: np.ndarray

    @property
    def rgb(self):
        return self.mask[:, :, :3]

    @property
    def rgba(self):
        return self.mask

    def labels(self):
        return self.mask[:, :, 0]


@dataclass
class SemanticSegmentation3D(Annotation):
    mask: np.ndarray


@dataclass
class BoundingBoxes3D(Annotation):
    boxes: List[BoundingBox3D]


@dataclass
class BoundingBox3D:
    pose: AnnotationPose
    width: float
    height: float
    length: float
    class_id: int
    class_name: str
    instance_id: int
    num_points: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        rep = f"Class ID: {self.class_id} {self.pose}"
        return rep

    @property
    def size(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height])  # assuming FLU

    @property
    def position(self) -> np.ndarray:
        return self.pose.translation


AnnotationType = Type[Annotation]


class AnnotationTypes:
    BoundingBox2D: Type[BoundingBox2D] = BoundingBox2D
    BoundingBoxes3D: Type[BoundingBoxes3D] = BoundingBoxes3D
    SemanticSegmentation2D: Type[SemanticSegmentation2D] = SemanticSegmentation2D
    SemanticSegmentation3D: Type[SemanticSegmentation3D] = SemanticSegmentation3D
