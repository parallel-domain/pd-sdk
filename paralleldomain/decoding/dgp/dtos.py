from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from dataclasses_json import CatchAll, DataClassJsonMixin, Undefined, config, dataclass_json


@dataclass_json
@dataclass
class TranslationDTO(DataClassJsonMixin):
    x: float
    y: float
    z: float


@dataclass_json
@dataclass
class RotationDTO(DataClassJsonMixin):
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass_json
@dataclass
class PoseDTO(DataClassJsonMixin):
    translation: TranslationDTO
    rotation: RotationDTO


@dataclass_json
@dataclass
class IdDTO(DataClassJsonMixin):
    timestamp: str  # TODO: Read as proper datetime object
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
class SceneDataDatumTypeGeneric(DataClassJsonMixin):
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
class SceneDataDatum(DataClassJsonMixin):
    image: Optional[SceneDataDatumTypeImage] = None
    point_cloud: Optional[SceneDataDatumTypePointCloud] = None


@dataclass_json
@dataclass
class SceneMetadataPDDTO(DataClassJsonMixin):
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
class SceneMetadataDTO(DataClassJsonMixin):
    PD: SceneMetadataPDDTO
    other: CatchAll


@dataclass_json
@dataclass
class SceneDataDTO(DataClassJsonMixin):
    next_key: str
    datum: SceneDataDatum
    id: SceneDataIdDTO
    key: str
    prev_key: str


@dataclass_json
@dataclass
class SceneSampleDTO(DataClassJsonMixin):
    calibration_key: str
    id: SceneSampleIdDTO
    datum_keys: List[str]
    metadata: Dict[str, Any]


@dataclass_json
@dataclass
class SceneDTO(DataClassJsonMixin):
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
class CalibrationIntrinsicDTO(DataClassJsonMixin):
    cx: float
    cy: float
    fx: float
    fy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    skew: float = 0.0
    fov: float = 0.0
    fisheye: Union[bool, int] = 0


@dataclass_json
@dataclass
class CalibrationDTO(DataClassJsonMixin):
    extrinsics: List[CalibrationExtrinsicDTO]
    names: List[str]
    intrinsics: List[CalibrationIntrinsicDTO]


@dataclass_json
@dataclass
class BoundingBox3DBoxDTO(DataClassJsonMixin):
    pose: PoseDTO
    width: float
    length: float
    height: float
    occlusion: float
    truncation: float


@dataclass_json
@dataclass
class BoundingBox3DDTO(DataClassJsonMixin):
    class_id: int
    instance_id: int
    num_points: int
    box: BoundingBox3DBoxDTO
    attributes: Dict[str, Any]


@dataclass_json
@dataclass
class AnnotationsDTO(DataClassJsonMixin):
    annotations: List[Any]


@dataclass_json
@dataclass
class AnnotationsBoundingBox3DDTO(AnnotationsDTO):
    annotations: List[BoundingBox3DDTO]


@dataclass_json
@dataclass
class BoundingBox2DBoxDTO(DataClassJsonMixin):
    x: int
    y: int
    w: int
    h: int


@dataclass_json
@dataclass
class BoundingBox2DDTO(DataClassJsonMixin):
    class_id: int
    instance_id: int
    iscrowd: bool
    box: BoundingBox2DBoxDTO
    attributes: Dict[str, Any]


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
class DatasetDTO:
    meta_data: DatasetMetaDTO
    scene_names: List[str]


@dataclass
class OntologyItemColorDTO:
    r: int
    g: int
    b: int


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
