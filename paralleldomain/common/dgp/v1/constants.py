from typing import Dict, Type, TypeVar

from paralleldomain.model.annotation import Annotation, AnnotationTypes
from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, CoordinateSystem
from paralleldomain.utilities.transformation import Transformation

TransformType = TypeVar("TransformType", bound=Transformation)
DGP_TO_INTERNAL_CS = CoordinateSystem("FLU") > INTERNAL_COORDINATE_SYSTEM

ANNOTATION_TYPE_MAP: Dict[int, Type[Annotation]] = {
    0: AnnotationTypes.BoundingBoxes2D,
    1: AnnotationTypes.BoundingBoxes3D,
    2: AnnotationTypes.SemanticSegmentation2D,
    3: AnnotationTypes.SemanticSegmentation3D,
    4: AnnotationTypes.InstanceSegmentation2D,
    5: AnnotationTypes.InstanceSegmentation3D,
    6: AnnotationTypes.Depth,
    7: AnnotationTypes.SurfaceNormals3D,
    8: AnnotationTypes.OpticalFlow,
    9: AnnotationTypes.SceneFlow,
    10: AnnotationTypes.Points2D,
    11: AnnotationTypes.Polylines2D,
    12: AnnotationTypes.Polygons2D,
    13: AnnotationTypes.SurfaceNormals2D,
    16: AnnotationTypes.Points3D,
    17: AnnotationTypes.Polylines3D,
    18: AnnotationTypes.Polygons3D,
    95: AnnotationTypes.MaterialProperties3D,
    96: AnnotationTypes.BackwardOpticalFlow,
    97: AnnotationTypes.BackwardSceneFlow,
    98: AnnotationTypes.Albedo2D,  # DGPv0 12
    99: AnnotationTypes.MaterialProperties2D,  # DGPv0 13
}

ANNOTATION_TYPE_MAP_INV: Dict[Type[Annotation], str] = {
    v: k for k, v in ANNOTATION_TYPE_MAP.items() if v is not Annotation
}

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


class PointFormat:
    X: str = "X"
    Y: str = "Y"
    Z: str = "Z"
    I: str = "INTENSITY"  # noqa: E741
    R: str = "R"
    G: str = "G"
    B: str = "B"
    RING: str = "RING"
    RAYTYPE: str = "RAYTYPE"
    TS: str = "TIMESTAMP"

    @classmethod
    def to_list(cls):
        return [
            cls.X,
            cls.Y,
            cls.Z,
            cls.I,
            cls.R,
            cls.G,
            cls.B,
            cls.RING,
            # cls.RAYTYPE,  # deactivated until in official DGP v1.0 proto schema
            cls.TS,
        ]


class RadarPointFormat:
    X: str = "X"
    Y: str = "Y"
    Z: str = "Z"
    POWER: str = "REFLECTED_POWER_DB"
    RCS: str = "RCS_DBSM"
    DOPPLER: str = "VELOCITY_XS"
    RANGE: str = "RADIAL_DISTANCE"
    AZ: str = "AZIMUTH_ANGLE"
    EL: str = "ELEVATION_ANGLE"
    TS: str = "TIMESTAMP"

    @classmethod
    def to_list(cls):
        return [
            cls.X,
            cls.Y,
            cls.Z,
            cls.POWER,
            cls.DOPPLER,
            cls.RANGE,
            cls.AZ,
            cls.EL,
            cls.TS,
        ]


class PointFormatDtype:
    X: str = "<f4"
    Y: str = "<f4"
    Z: str = "<f4"
    I: str = "<f4"  # noqa: E741
    R: str = "<f4"
    G: str = "<f4"
    B: str = "<f4"
    RING: str = "<u4"
    RAYTYPE: str = "<u4"
    TS: str = "<u8"

    @classmethod
    def to_list(cls):
        return [
            cls.X,
            cls.Y,
            cls.Z,
            cls.I,
            cls.R,
            cls.G,
            cls.B,
            cls.RING,
            # cls.RAYTYPE,  # deactivated until in official DGP v1.0 proto schema
            cls.TS,
        ]


class RadarPointFormatDtype:
    X: str = "<f4"
    Y: str = "<f4"
    Z: str = "<f4"
    POWER: str = "<f4"
    DOPPLER: str = "<f4"
    RANGE: str = "<f4"
    AZ: str = "<f4"
    EL: str = "<f4"
    TS: str = "<u8"

    @classmethod
    def to_list(cls):
        return [
            cls.X,
            cls.Y,
            cls.Z,
            cls.POWER,
            cls.DOPPLER,
            cls.RANGE,
            cls.AZ,
            cls.EL,
            cls.TS,
        ]


class DirectoryName:
    ALBEDO_2D: str = "base_color"
    BOUNDING_BOX_2D: str = "bounding_box_2d"
    BOUNDING_BOX_3D: str = "bounding_box_3d"
    CALIBRATION: str = "calibration"
    DEPTH: str = "depth"
    INSTANCE_SEGMENTATION_2D: str = "instance_segmentation_2d"
    INSTANCE_SEGMENTATION_3D: str = "instance_segmentation_3d"
    KEY_LINE_2D: str = "key_line_2d"
    KEY_LINE_3D: str = "key_line_3d"
    KEY_POINT_2D: str = "key_point_2d"
    KEY_POINT_3D: str = "key_point_3d"
    MATERIAL_PROPERTIES_2D: str = "roughness_metallic_specular"
    MATERIAL_PROPERTIES_3D: str = "material_properties_3d"
    MOTION_VECTORS_2D: str = "motion_vectors_2d"
    MOTION_VECTORS_3D: str = "surface_properties_3d"
    BACK_MOTION_VECTORS_2D: str = "back_motion_vectors_2d"
    BACK_MOTION_VECTORS_3D: str = "back_motion_vectors_3d"
    ONTOLOGY: str = "ontology"
    POINT_CLOUD: str = "point_cloud"
    POLYGON_2D: str = "polygon_2d"
    POLYGON_3D: str = "polygon_3d"
    RADAR_POINT_CLOUD: str = "radar_point_cloud"
    RGB: str = "rgb"
    SEMANTIC_SEGMENTATION_2D: str = "semantic_segmentation_2d"
    SEMANTIC_SEGMENTATION_3D: str = "semantic_segmentation_3d"
    SURFACE_NORMALS_2D: str = "surface_normals_2d"
    SURFACE_NORMALS_3D: str = "surface_normals_3d"
