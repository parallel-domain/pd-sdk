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
    "6": AnnotationTypes.Depth,
    "7": AnnotationTypes.SurfaceNormals3D,
    "8": AnnotationTypes.OpticalFlow,
    "9": AnnotationTypes.SceneFlow,
    "10": AnnotationTypes.SurfaceNormals2D,
    "12": AnnotationTypes.Albedo2D,
    "13": AnnotationTypes.MaterialProperties2D,
    "15": AnnotationTypes.MaterialProperties3D,
}

ANNOTATION_TYPE_MAP_INV: Dict[Type[Annotation], str] = {
    v: k for k, v in ANNOTATION_TYPE_MAP.items() if v is not Annotation
}

POINT_FORMAT = ("X", "Y", "Z", "INTENSITY", "R", "G", "B", "RING", "TIMESTAMP")
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

NON_DGP_ANNOTATIONS = {
    AnnotationTypes.Albedo2D,
    AnnotationTypes.MaterialProperties2D,
    AnnotationTypes.MaterialProperties3D,
}


class DirectoryName:
    ALBEDO_2D: str = "base_color"
    BOUNDING_BOX_2D: str = "bounding_box_2d"
    BOUNDING_BOX_3D: str = "bounding_box_3d"
    CALIBRATION: str = "calibration"
    DEPTH: str = "depth"
    INSTANCE_SEGMENTATION_2D: str = "instance_segmentation_2d"
    INSTANCE_SEGMENTATION_3D: str = "instance_segmentation_3d"
    MATERIAL_PROPERTIES_2D: str = "roughness_metallic_specular"
    MATERIAL_PROPERTIES_3D: str = "material_properties_3d"
    MOTION_VECTORS_2D: str = "motion_vectors_2d"
    MOTION_VECTORS_3D: str = "motion_vectors_3d"
    ONTOLOGY: str = "ontology"
    POINT_CLOUD: str = "point_cloud"
    RGB: str = "rgb"
    SEMANTIC_SEGMENTATION_2D: str = "semantic_segmentation_2d"
    SEMANTIC_SEGMENTATION_3D: str = "semantic_segmentation_3d"
    SURFACE_NORMALS_2D: str = "surface_normals_2d"
    SURFACE_NORMALS_3D: str = "surface_normals_3d"
