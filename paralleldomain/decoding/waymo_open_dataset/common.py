import json
import struct
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Tuple

from paralleldomain.decoding.common import LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from paralleldomain.decoding.waymo_open_dataset.protos import dataset_pb2
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache

SEGMENTATION_COLOR_MAP = {
    cs_pb2.CameraSegmentation.TYPE_UNDEFINED: dict(r=0, g=0, b=0),
    cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE: dict(r=102, g=102, b=102),
    cs_pb2.CameraSegmentation.TYPE_CAR: dict(r=0, g=0, b=142),
    cs_pb2.CameraSegmentation.TYPE_TRUCK: dict(r=0, g=0, b=70),
    cs_pb2.CameraSegmentation.TYPE_BUS: dict(r=0, g=60, b=100),
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE: dict(r=61, g=133, b=198),
    cs_pb2.CameraSegmentation.TYPE_BICYCLE: dict(r=119, g=11, b=32),
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE: dict(r=0, g=0, b=230),
    cs_pb2.CameraSegmentation.TYPE_TRAILER: dict(r=111, g=168, b=220),
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN: dict(r=220, g=20, b=60),
    cs_pb2.CameraSegmentation.TYPE_CYCLIST: dict(r=255, g=0, b=0),
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST: dict(r=180, g=0, b=0),
    cs_pb2.CameraSegmentation.TYPE_BIRD: dict(r=127, g=96, b=0),
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL: dict(r=91, g=15, b=0),
    cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE: dict(r=230, g=145, b=56),
    cs_pb2.CameraSegmentation.TYPE_POLE: dict(r=153, g=153, b=153),
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT: dict(r=234, g=153, b=153),
    cs_pb2.CameraSegmentation.TYPE_SIGN: dict(r=246, g=178, b=107),
    cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT: dict(r=250, g=170, b=30),
    cs_pb2.CameraSegmentation.TYPE_BUILDING: dict(r=70, g=70, b=70),
    cs_pb2.CameraSegmentation.TYPE_ROAD: dict(r=128, g=64, b=128),
    cs_pb2.CameraSegmentation.TYPE_LANE_MARKER: dict(r=234, g=209, b=220),
    cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER: dict(r=217, g=210, b=233),
    cs_pb2.CameraSegmentation.TYPE_SIDEWALK: dict(r=244, g=35, b=232),
    cs_pb2.CameraSegmentation.TYPE_VEGETATION: dict(r=107, g=142, b=35),
    cs_pb2.CameraSegmentation.TYPE_SKY: dict(r=70, g=130, b=180),
    cs_pb2.CameraSegmentation.TYPE_GROUND: dict(r=102, g=102, b=102),
    cs_pb2.CameraSegmentation.TYPE_DYNAMIC: dict(r=102, g=102, b=102),
    cs_pb2.CameraSegmentation.TYPE_STATIC: dict(r=102, g=102, b=102),
}


WAYMO_INDEX_TO_CAMERA_NAME = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "SIDE_LEFT",
    4: "FRONT_RIGHT",
    5: "SIDE_RIGHT",
}

WAYMO_CAMERA_NAME_TO_INDEX = {v: k for k, v in WAYMO_INDEX_TO_CAMERA_NAME.items()}

WAYMO_USE_ALL_LIDAR_NAME = "all"

WAYMO_INDEX_TO_LIDAR_NAME = {0: "UNKNOWN", 1: "TOP", 2: "FRONT", 3: "SIDE_LEFT", 4: "SIDE_RIGHT", 5: "REAR"}

WAYMO_LIDAR_NAME_TO_INDEX = {v: k for k, v in WAYMO_INDEX_TO_LIDAR_NAME.items()}


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


def load_pre_calculated_scene_to_id_map(
    split_name: str, index_folder: AnyPath
) -> Dict[SceneName, List[Dict[str, Any]]]:
    file_path = index_folder / f"{split_name}_scenes_to_frame_info.json"
    with file_path.open("r") as fp:
        return json.load(fp)


def get_cached_pre_calculated_scene_to_frame_info(
    lazy_load_cache: LazyLoadCache, dataset_name: str, split_name: str, index_folder: AnyPath
) -> Dict[SceneName, List[Dict[str, Any]]]:
    _unique_cache_key = create_cache_key(dataset_name=dataset_name, extra=f"{split_name}scene_to_fid")
    id_map = lazy_load_cache.get_item(
        key=_unique_cache_key,
        loader=partial(load_pre_calculated_scene_to_id_map, split_name=split_name, index_folder=index_folder),
    )
    return id_map


def load_pre_calculated_scene_to_has_annotation(
    split_name: str, annotation_str: str, index_folder: AnyPath
) -> Dict[str, bool]:
    file_path = index_folder / f"{split_name}_sensor_frame_to_has_{annotation_str}.json"
    with file_path.open("r") as fp:
        return json.load(fp)


def get_cached_pre_calculated_scene_to_has_annotation(
    lazy_load_cache: LazyLoadCache,
    dataset_name: str,
    scene_name: SceneName,
    frame_id: FrameId,
    sensor_name: SensorName,
    split_name: str,
    annotation_type: AnnotationType,
    index_folder: AnyPath,
) -> bool:
    annotation_str = {
        AnnotationTypes.SemanticSegmentation2D: "segmentation",
        AnnotationTypes.InstanceSegmentation2D: "segmentation",
        AnnotationTypes.BoundingBoxes2D: "bounding_box_2d",
    }[annotation_type]

    _unique_cache_key = create_cache_key(
        dataset_name=dataset_name, extra=f"{split_name}_sensor_frame_to_has_{annotation_str}"
    )
    id_map = lazy_load_cache.get_item(
        key=_unique_cache_key,
        loader=partial(
            load_pre_calculated_scene_to_has_annotation,
            split_name=split_name,
            annotation_str=annotation_str,
            index_folder=index_folder,
        ),
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
            key=_unique_cache_key, loader=lambda: get_record_at(frame_id=frame_id, record_path=self.record_path)
        )


WAYMO_SEMSEG_CLASSES = [
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

WAYMO_3DBB_CLASSES = [
    ClassDetail(name="UNDEFINED", id=0, instanced=True, meta=dict()),
    ClassDetail(name="VEHICLE", id=1, instanced=True, meta=dict()),
    ClassDetail(name="PEDESTRIAN", id=2, instanced=True, meta=dict()),
    ClassDetail(name="SIGN", id=3, instanced=True, meta=dict()),
    ClassDetail(name="CYCLIST", id=4, instanced=True, meta=dict()),
]

WAYMO_2DBB_CLASSES = [
    ClassDetail(name="VEHICLE", id=1, instanced=True, meta=dict()),
    ClassDetail(name="PEDESTRIAN", id=2, instanced=True, meta=dict()),
    ClassDetail(name="CYCLIST", id=4, instanced=True, meta=dict()),
]


def decode_class_maps() -> Dict[AnnotationIdentifier, ClassMap]:
    return {
        AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D): ClassMap(
            classes=WAYMO_SEMSEG_CLASSES
        ),
        AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D): ClassMap(classes=WAYMO_3DBB_CLASSES),
        AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D): ClassMap(classes=WAYMO_2DBB_CLASSES),
    }
