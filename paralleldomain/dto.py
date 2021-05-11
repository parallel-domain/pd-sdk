from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, Undefined, CatchAll, config
from typing import List, Dict, Any, Optional
from .utils import Transformation
from .sensor import SensorExtrinsic


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
class SceneDataDatum:
    image: Optional[SceneDataDatumTypeImage] = None
    point_cloud: Optional[SceneDataDatumTypePointCloud] = None


@dataclass_json
@dataclass
class SceneMetadataPDDTO:
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
class CalibrationDTO:
    extrinsics: List[CalibrationExtrinsicDTO]
    names: List[str]
    intrinsics: List[CalibrationIntrinsicDTO]
