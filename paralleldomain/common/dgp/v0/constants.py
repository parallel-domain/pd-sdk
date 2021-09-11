from typing import Dict, Type, TypeVar

from paralleldomain.model.annotation import Annotation, AnnotationTypes
from paralleldomain.model.transformation import Transformation
from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, CoordinateSystem

TransformType = TypeVar("TransformType", bound=Transformation)
DGP_TO_INTERNAL_CS = CoordinateSystem("FLU") > INTERNAL_COORDINATE_SYSTEM

ANNOTATION_TYPE_MAP: Dict[str, Type[Annotation]] = {
    "0": AnnotationTypes.BoundingBoxes2D,
    "1": AnnotationTypes.BoundingBoxes3D,
    "2": AnnotationTypes.SemanticSegmentation2D,
    "3": AnnotationTypes.SemanticSegmentation3D,
    "4": AnnotationTypes.InstanceSegmentation2D,
    "5": AnnotationTypes.InstanceSegmentation3D,
    "6": AnnotationTypes.Depth,  # Depth
    "7": Annotation,  # Surface Normals 3D
    "8": AnnotationTypes.OpticalFlow,
    "9": Annotation,  # Motion Vectors 3D aka Scene Flow
    "10": Annotation,  # Surface normals 2D
    "12": Annotation,  # Albedo
    "13": Annotation,  # Material Properties
}

ANNOTATION_TYPE_MAP_INV: Dict[Type[Annotation], str] = {
    v: k for k, v in ANNOTATION_TYPE_MAP.items() if v is not Annotation
}
