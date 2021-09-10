from typing import Dict, Type, TypeVar

from paralleldomain.model.annotation import Annotation, AnnotationTypes
from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, CoordinateSystem
from paralleldomain.utilities.transformation import Transformation

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
}

ANNOTATION_TYPE_MAP_INV: Dict[Type[Annotation], str] = {
    v: k for k, v in ANNOTATION_TYPE_MAP.items() if v is not Annotation
}

POINT_FORMAT = ("X", "Y", "Z", "INTENSITY", "R", "G", "B", "RING", "TIMESTAMP")
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


class DirectoryName:
    CALIBRATION: str = "calibration"
    ONTOLOGY: str = "ontology"
    RGB: str = "rgb"
    POINT_CLOUD: str = "point_cloud"
    BOUNDING_BOX_2D: str = "bounding_box_2d"
    BOUNDING_BOX_3D: str = "bounding_box_3d"
    SEMANTIC_SEGMENTATION_2D: str = "semantic_segmentation_2d"
    INSTANCE_SEGMENTATION_2D: str = "instance_segmentation_2d"
    SEMANTIC_SEGMENTATION_3D: str = "semantic_segmentation_3d"
    INSTANCE_SEGMENTATION_3D: str = "instance_segmentation_3d"
    MOTION_VECTORS_2D: str = "motion_vectors_2d"
    DEPTH: str = "depth"
