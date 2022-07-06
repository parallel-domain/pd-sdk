import json
import struct
from typing import Any, Dict, Generator, List, Optional, Tuple

from paralleldomain.decoding.common import LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from paralleldomain.decoding.waymo_open_dataset.protos import dataset_pb2
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache

SEGMENTATION_COLOR_MAP = {
    cs_pb2.CameraSegmentation.TYPE_UNDEFINED: [0, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_CAR: [0, 0, 142],
    cs_pb2.CameraSegmentation.TYPE_TRUCK: [0, 0, 70],
    cs_pb2.CameraSegmentation.TYPE_BUS: [0, 60, 100],
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE: [61, 133, 198],
    cs_pb2.CameraSegmentation.TYPE_BICYCLE: [119, 11, 32],
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE: [0, 0, 230],
    cs_pb2.CameraSegmentation.TYPE_TRAILER: [111, 168, 220],
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN: [220, 20, 60],
    cs_pb2.CameraSegmentation.TYPE_CYCLIST: [255, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST: [180, 0, 0],
    cs_pb2.CameraSegmentation.TYPE_BIRD: [127, 96, 0],
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL: [91, 15, 0],
    cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE: [230, 145, 56],
    cs_pb2.CameraSegmentation.TYPE_POLE: [153, 153, 153],
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT: [234, 153, 153],
    cs_pb2.CameraSegmentation.TYPE_SIGN: [246, 178, 107],
    cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT: [250, 170, 30],
    cs_pb2.CameraSegmentation.TYPE_BUILDING: [70, 70, 70],
    cs_pb2.CameraSegmentation.TYPE_ROAD: [128, 64, 128],
    cs_pb2.CameraSegmentation.TYPE_LANE_MARKER: [234, 209, 220],
    cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER: [217, 210, 233],
    cs_pb2.CameraSegmentation.TYPE_SIDEWALK: [244, 35, 232],
    cs_pb2.CameraSegmentation.TYPE_VEGETATION: [107, 142, 35],
    cs_pb2.CameraSegmentation.TYPE_SKY: [70, 130, 180],
    cs_pb2.CameraSegmentation.TYPE_GROUND: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_DYNAMIC: [102, 102, 102],
    cs_pb2.CameraSegmentation.TYPE_STATIC: [102, 102, 102],
}


WAYMO_INDEX_TO_CAMERA_NAME = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

WAYMO_CAMERA_NAME_TO_INDEX = {v: k for k, v in WAYMO_INDEX_TO_CAMERA_NAME.items()}


def get_record_iterator(
    record_path: AnyPath, read_frame: bool
) -> Generator[Tuple[Optional[dataset_pb2.Frame], FrameId], None, None]:
    with record_path.open("rb") as file:
        while file:
            frame_id = file.tell()
            header = file.read(12)
            if header == b"":
                break
            length, lengthcrc = struct.unpack("QI", header)
            if lengthcrc == 0:
                break
            frame = None
            if read_frame:
                data = file.read(length)
                _ = struct.unpack("I", file.read(4))

                frame = dataset_pb2.Frame()
                frame.ParseFromString(data)
            else:
                file.seek(length + 4, 1)

            yield frame, str(frame_id)


def get_record_at(record_path: AnyPath, frame_id: FrameId) -> Optional[dataset_pb2.Frame]:
    with record_path.open("rb") as file:
        file.seek(int(frame_id), 0)
        read_frame_id = file.tell()
        assert str(read_frame_id) == frame_id
        header = file.read(12)
        if header == b"":
            return None
        length, lengthcrc = struct.unpack("QI", header)

        data = file.read(length)

        frame = dataset_pb2.Frame()
        frame.ParseFromString(data)

        return frame


def load_pre_calcualted_train_scene_to_id_map() -> Dict[SceneName, List[Dict[str, Any]]]:
    file_path = AnyPath(__file__).absolute().parent / "pre_calculated" / "train_scenes_to_frame_info.json"
    with file_path.open("r") as fp:
        return json.load(fp)


def get_cached_pre_calcualted_train_scene_to_id_map(
    lazy_load_cache: LazyLoadCache, dataset_name: str
) -> Dict[SceneName, List[Dict[str, Any]]]:
    _unique_cache_key = create_cache_key(dataset_name=dataset_name, extra="training_scene_to_fid")
    id_map = lazy_load_cache.get_item(
        key=_unique_cache_key,
        loader=load_pre_calcualted_train_scene_to_id_map,
    )
    return id_map


def load_pre_calcualted_train_scene_to_has_segmentation() -> Dict[str, bool]:
    file_path = AnyPath(__file__).absolute().parent / "pre_calculated" / "train_sensor_frame_to_has_segmentation.json"
    with file_path.open("r") as fp:
        return json.load(fp)


def get_cached_pre_calcualted_train_scene_to_has_segmentation(
    lazy_load_cache: LazyLoadCache, dataset_name: str, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName
) -> bool:
    _unique_cache_key = create_cache_key(dataset_name=dataset_name, extra="train_sensor_frame_to_has_segmentation")
    id_map = lazy_load_cache.get_item(
        key=_unique_cache_key,
        loader=load_pre_calcualted_train_scene_to_has_segmentation,
    )
    key = f"{scene_name}-{frame_id}-{sensor_name}"
    if key in id_map:
        return id_map[key]
    return False


class WaymoFileAccessMixin(LazyLoadPropertyMixin):
    def __init__(self, record_path: AnyPath):
        self.record_path = record_path

    def get_record_at(self, frame_id: FrameId) -> Optional[dataset_pb2.Frame]:
        _unique_cache_key = create_cache_key(
            dataset_name="Waymo Open Dataset", scene_name=self.record_path.name, frame_id=frame_id, extra="record_data"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: get_record_at(frame_id=frame_id, record_path=self.record_path),
        )


WAYMO_CLASSES = [
    ClassDetail(
        name="UNDEFINED",
        id=cs_pb2.CameraSegmentation.TYPE_UNDEFINED,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_UNDEFINED]),
    ),
    ClassDetail(
        name="EGO_VEHICLE",
        id=cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE]),
    ),
    ClassDetail(
        name="CAR",
        id=cs_pb2.CameraSegmentation.TYPE_CAR,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_CAR]),
    ),
    ClassDetail(
        name="TRUCK",
        id=cs_pb2.CameraSegmentation.TYPE_TRUCK,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_TRUCK]),
    ),
    ClassDetail(
        name="BUS",
        id=cs_pb2.CameraSegmentation.TYPE_BUS,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_BUS]),
    ),
    ClassDetail(
        name="OTHER_LARGE_VEHICLE",
        id=cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE]),
    ),
    ClassDetail(
        name="BICYCLE",
        id=cs_pb2.CameraSegmentation.TYPE_BICYCLE,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_BICYCLE]),
    ),
    ClassDetail(
        name="MOTORCYCLE",
        id=cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE]),
    ),
    ClassDetail(
        name="TRAILER",
        id=cs_pb2.CameraSegmentation.TYPE_TRAILER,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_TRAILER]),
    ),
    ClassDetail(
        name="PEDESTRIAN",
        id=cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN]),
    ),
    ClassDetail(
        name="CYCLIST",
        id=cs_pb2.CameraSegmentation.TYPE_CYCLIST,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_CYCLIST]),
    ),
    ClassDetail(
        name="MOTORCYCLIST",
        id=cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST]),
    ),
    ClassDetail(
        name="BIRD",
        id=cs_pb2.CameraSegmentation.TYPE_BIRD,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_BIRD]),
    ),
    ClassDetail(
        name="GROUND_ANIMAL",
        id=cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL]),
    ),
    ClassDetail(
        name="CONSTRUCTION_CONE_POLE",
        id=cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE]),
    ),
    ClassDetail(
        name="POLE",
        id=cs_pb2.CameraSegmentation.TYPE_POLE,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_POLE]),
    ),
    ClassDetail(
        name="PEDESTRIAN_OBJECT",
        id=cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT]),
    ),
    ClassDetail(
        name="SIGN",
        id=cs_pb2.CameraSegmentation.TYPE_SIGN,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_SIGN]),
    ),
    ClassDetail(
        name="TRAFFIC_LIGHT",
        id=cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT]),
    ),
    ClassDetail(
        name="BUILDING",
        id=cs_pb2.CameraSegmentation.TYPE_BUILDING,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_BUILDING]),
    ),
    ClassDetail(
        name="ROAD",
        id=cs_pb2.CameraSegmentation.TYPE_ROAD,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_ROAD]),
    ),
    ClassDetail(
        name="LANE_MARKER",
        id=cs_pb2.CameraSegmentation.TYPE_LANE_MARKER,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_LANE_MARKER]),
    ),
    ClassDetail(
        name="ROAD_MARKER",
        id=cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER]),
    ),
    ClassDetail(
        name="SIDEWALK",
        id=cs_pb2.CameraSegmentation.TYPE_SIDEWALK,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_SIDEWALK]),
    ),
    ClassDetail(
        name="VEGETATION",
        id=cs_pb2.CameraSegmentation.TYPE_VEGETATION,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_VEGETATION]),
    ),
    ClassDetail(
        name="SKY",
        id=cs_pb2.CameraSegmentation.TYPE_SKY,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_SKY]),
    ),
    ClassDetail(
        name="GROUND",
        id=cs_pb2.CameraSegmentation.TYPE_GROUND,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_GROUND]),
    ),
    ClassDetail(
        name="DYNAMIC",
        id=cs_pb2.CameraSegmentation.TYPE_DYNAMIC,
        instanced=True,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_DYNAMIC]),
    ),
    ClassDetail(
        name="STATIC",
        id=cs_pb2.CameraSegmentation.TYPE_STATIC,
        instanced=False,
        meta=dict(color=SEGMENTATION_COLOR_MAP[cs_pb2.CameraSegmentation.TYPE_STATIC]),
    ),
]


def decode_class_maps() -> Dict[AnnotationType, ClassMap]:
    return {AnnotationTypes.SemanticSegmentation2D: ClassMap(classes=WAYMO_CLASSES)}
