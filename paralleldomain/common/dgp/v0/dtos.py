from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import ujson
from dataclasses_json import CatchAll, Undefined, config, dataclass_json

from paralleldomain.model.annotation import BoundingBox2D, BoundingBox3D
from paralleldomain.model.class_mapping import ClassMap


def _attribute_key_dump(obj: object) -> str:
    return str(obj)


def _attribute_value_dump(obj: object) -> str:
    if isinstance(obj, Dict) or isinstance(obj, List):
        return ujson.dumps(obj, indent=2, escape_forward_slashes=False)
    else:
        return str(obj)


@dataclass_json
@dataclass
class TranslationDTO:
    x: float
    y: float
    z: float


@dataclass_json
@dataclass
class RotationDTO:
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass_json
@dataclass
class PoseDTO:
    translation: TranslationDTO
    rotation: RotationDTO


@dataclass_json
@dataclass
class IdDTO:
    timestamp: str
    index: str
    log: str
    name: str


@dataclass_json
@dataclass
class SceneDataIdDTO(IdDTO):
    ...


@dataclass_json
@dataclass
class SceneSampleIdDTO(IdDTO):
    ...


@dataclass_json
@dataclass
class SceneDataDatumTypeGeneric:
    pose: PoseDTO
    filename: str
    annotations: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass_json
@dataclass
class SceneDataDatumTypeImage(SceneDataDatumTypeGeneric):
    height: int
    width: int
    channels: int


@dataclass_json
@dataclass
class SceneDataDatumTypePointCloud(SceneDataDatumTypeGeneric):
    point_fields: List[Any]
    point_format: List[str]


@dataclass_json
@dataclass
class SceneDataDatumImage:
    image: Optional[SceneDataDatumTypeImage] = None


@dataclass_json
@dataclass
class SceneDataDatumPointCloud:
    point_cloud: Optional[SceneDataDatumTypePointCloud] = None


@dataclass_json
@dataclass
class SceneDataDatum(SceneDataDatumImage, SceneDataDatumPointCloud):
    ...


@dataclass_json
@dataclass
class SceneMetadataPDDTO:
    type_: str = field(metadata=config(field_name="@type"))
    location: str
    time_of_day: str
    version: int
    cloud_cover: float
    sun_elevation: float
    sun_azimuth: float
    fog_intensity: float
    rain_intensity: float
    wetness: float
    street_lights: float
    batch_id: int
    region_type: str
    scene_type: Optional[str] = None


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SceneMetadataDTO:
    PD: SceneMetadataPDDTO
    other: CatchAll


@dataclass_json
@dataclass
class SceneDataDTO:
    next_key: str
    datum: SceneDataDatum
    id: SceneDataIdDTO
    key: str
    prev_key: str


@dataclass_json
@dataclass
class SceneSampleDTO:
    calibration_key: str
    id: SceneSampleIdDTO
    datum_keys: List[str]
    metadata: Dict[str, Any]


@dataclass_json
@dataclass
class SceneDTO:
    name: str
    description: str
    log: str
    data: List[SceneDataDTO]
    ontologies: Dict[str, str]
    metadata: SceneMetadataDTO
    samples: List[SceneSampleDTO]


@dataclass_json
@dataclass
class CalibrationExtrinsicDTO(PoseDTO):
    ...


@dataclass_json
@dataclass
class CalibrationIntrinsicDTO:
    cx: float
    cy: float
    fx: float
    fy: float
    skew: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    fov: float = 0.0
    fisheye: Union[bool, int] = 0


@dataclass_json
@dataclass
class CalibrationDTO:
    extrinsics: List[CalibrationExtrinsicDTO]
    names: List[str]
    intrinsics: List[CalibrationIntrinsicDTO]


@dataclass_json
@dataclass
class BoundingBox3DBoxDTO:
    pose: PoseDTO
    width: float
    length: float
    height: float
    occlusion: int
    truncation: float


@dataclass_json
@dataclass
class BoundingBox3DDTO:
    class_id: int
    instance_id: int
    num_points: int
    box: BoundingBox3DBoxDTO
    attributes: Dict[str, Any]

    @staticmethod
    def from_bounding_box(box: BoundingBox3D) -> "BoundingBox3DDTO":
        try:
            occlusion = box.attributes["occlusion"]
            del box.attributes["occlusion"]
        except KeyError:
            occlusion = 0

        try:
            truncation = box.attributes["truncation"]
            del box.attributes["truncation"]
        except KeyError:
            truncation = 0

        box_dto = BoundingBox3DDTO(
            class_id=box.class_id,
            instance_id=box.instance_id,
            num_points=box.num_points,
            attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in box.attributes.items()},
            box=BoundingBox3DBoxDTO(
                width=box.width,
                length=box.length,
                height=box.height,
                occlusion=occlusion,
                truncation=truncation,
                pose=PoseDTO(
                    translation=TranslationDTO(
                        x=box.pose.translation[0], y=box.pose.translation[1], z=box.pose.translation[2]
                    ),
                    rotation=RotationDTO(
                        qw=box.pose.quaternion.w,
                        qx=box.pose.quaternion.x,
                        qy=box.pose.quaternion.y,
                        qz=box.pose.quaternion.z,
                    ),
                ),
            ),
        )

        return box_dto


@dataclass_json
@dataclass
class AnnotationsDTO:
    annotations: List[Any]


@dataclass_json
@dataclass
class AnnotationsBoundingBox3DDTO(AnnotationsDTO):
    annotations: List[BoundingBox3DDTO]


@dataclass_json
@dataclass
class BoundingBox2DBoxDTO:
    x: int
    y: int
    w: int
    h: int


@dataclass_json
@dataclass
class BoundingBox2DDTO:
    class_id: int
    instance_id: int
    area: int
    iscrowd: bool
    box: BoundingBox2DBoxDTO
    attributes: Dict[str, Any]

    @staticmethod
    def from_bounding_box(box: BoundingBox2D) -> "BoundingBox2DDTO":
        try:
            is_crowd = box.attributes["iscrowd"]
            del box.attributes["iscrowd"]
        except KeyError:
            is_crowd = False
        box_dto = BoundingBox2DDTO(
            class_id=box.class_id,
            instance_id=box.instance_id,
            area=box.area,
            iscrowd=is_crowd,
            attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in box.attributes.items()},
            box=BoundingBox2DBoxDTO(x=box.x, y=box.y, w=box.width, h=box.height),
        )

        return box_dto


@dataclass_json
@dataclass
class AnnotationsBoundingBox2DDTO(AnnotationsDTO):
    annotations: List[BoundingBox2DDTO]


@dataclass_json
@dataclass
class DatasetMetaDTO:
    origin: str
    name: str
    creator: str
    available_annotation_types: List[int]
    creation_date: str
    version: str
    description: str


@dataclass_json
@dataclass
class DatasetSceneSplitDTO:
    filenames: List[str]


@dataclass_json
@dataclass
class DatasetDTO:
    metadata: DatasetMetaDTO
    scene_splits: Dict[str, DatasetSceneSplitDTO]


@dataclass_json
@dataclass
class OntologyItemColorDTO:
    r: int
    g: int
    b: int


@dataclass_json
@dataclass
class OntologyItemDTO:
    name: str
    id: int
    color: OntologyItemColorDTO
    isthing: bool
    supercategory: str


@dataclass_json
@dataclass
class OntologyFileDTO:
    items: List[OntologyItemDTO]

    @staticmethod
    def from_class_map(class_map: ClassMap) -> "OntologyFileDTO":
        return OntologyFileDTO(
            items=[
                OntologyItemDTO(
                    id=cid,
                    name=cval.name,
                    color=OntologyItemColorDTO(
                        r=cval.meta["color"]["r"],
                        g=cval.meta["color"]["g"],
                        b=cval.meta["color"]["b"],
                    ),
                    isthing=cval.instanced,
                    supercategory="",
                )
                for cid, cval in class_map.items()
            ]
        )
