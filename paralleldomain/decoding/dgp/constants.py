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
}

ANNOTATION_TYPE_MAP_INV: Dict[Type[Annotation], str] = {
    v: k for k, v in ANNOTATION_TYPE_MAP.items() if v is not Annotation
}

"""
DGPLabel = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'Car', 'Person', ... .
        "id",  # An integer ID that is associated with this label.
        "is_thing",  # Whether this label distinguishes between single instances or not
    ],
)

_default_labels: List[DGPLabel] = [
    DGPLabel("Animal", 0, True),
    DGPLabel("Bicycle", 1, True),
    DGPLabel("Bicyclist", 2, True),
    DGPLabel("Building", 3, False),
    DGPLabel("Bus", 4, True),
    DGPLabel("Car", 5, True),
    DGPLabel("Caravan/RV", 6, True),
    DGPLabel("ConstructionVehicle", 7, True),
    DGPLabel("CrossWalk", 8, True),
    DGPLabel("Fence", 9, False),
    DGPLabel("HorizontalPole", 10, True),
    DGPLabel("LaneMarking", 11, False),
    DGPLabel("LimitLine", 12, False),
    DGPLabel("Motorcycle", 13, True),
    DGPLabel("Motorcyclist", 14, True),
    DGPLabel("OtherDriveableSurface", 15, False),
    DGPLabel("OtherFixedStructure", 16, False),
    DGPLabel("OtherMovable", 17, True),
    DGPLabel("OtherRider", 18, True),
    DGPLabel("Overpass/Bridge/Tunnel", 19, False),
    DGPLabel("OwnCar(EgoCar)", 20, False),
    DGPLabel("ParkingMeter", 21, False),
    DGPLabel("Pedestrian", 22, True),
    DGPLabel("Railway", 23, False),
    DGPLabel("Road", 24, False),
    DGPLabel("RoadBarriers", 25, False),
    DGPLabel("RoadBoundary(Curb)", 26, False),
    DGPLabel("RoadMarking", 27, False),
    DGPLabel("SideWalk", 28, False),
    DGPLabel("Sky", 29, False),
    DGPLabel("TemporaryConstructionObject", 30, True),
    DGPLabel("Terrain", 31, False),
    DGPLabel("TowedObject", 32, True),
    DGPLabel("TrafficLight", 33, True),
    DGPLabel("TrafficSign", 34, True),
    DGPLabel("Train", 35, True),
    DGPLabel("Truck", 36, True),
    DGPLabel("Vegetation", 37, False),
    DGPLabel("VerticalPole", 38, True),
    DGPLabel("WheeledSlow", 39, True),
    DGPLabel("LaneMarkingOther", 40, False),
    DGPLabel("LaneMarkingGap", 41, False),
    DGPLabel("Void", 255, False),
]

DEFAULT_CLASS_MAP = ClassMap(class_id_to_class_name={label.id: label.name for label in _default_labels})
"""
