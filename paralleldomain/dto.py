from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, Undefined, CatchAll, config, DataClassJsonMixin
from typing import List, Dict, Any, Optional


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
    region_type: str
    scene_type: str
    cloud_cover: float
    sun_elevation: float
    sun_azimuth: float
    fog_intensity: float
    rain_intensity: float
    wetness: float
    street_lights: float
    batch_id: int


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
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    k4: float
    k5: float
    k6: float
    skew: float
    fov: float
    fisheye: bool


@dataclass_json
@dataclass
class CalibrationDTO(DataClassJsonMixin):
    extrinsics: List[CalibrationExtrinsicDTO]
    names: List[str]
    intrinsics: List[CalibrationIntrinsicDTO]


@dataclass_json
@dataclass
class BoundingBox3DAttributesDTO(DataClassJsonMixin):
    vehicle_type: str
    point_cache: str
    parked_vehicle: Optional[str] = None


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
    attributes: BoundingBox3DAttributesDTO


@dataclass_json
@dataclass
class AnnotationsDTO(DataClassJsonMixin):
    annotations: List[Any]


@dataclass_json
@dataclass
class AnnotationsBoundingBox3DDTO(AnnotationsDTO):
    annotations: List[BoundingBox3DDTO]
