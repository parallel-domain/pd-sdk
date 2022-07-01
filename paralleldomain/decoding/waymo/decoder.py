import struct
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.decoding.waymo.common import decode_class_maps, get_record_iterator
from paralleldomain.decoding.waymo.frame_decoder import WaymoOpenDatasetFrameDecoder
from paralleldomain.decoding.waymo.sensor_decoder import WaymoOpenDatasetCameraSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

IMAGE_FOLDER_NAME = "image"
SEMANTIC_SEGMENTATION_FOLDER_NAME = "semantic_segmentation"
METADATA_FOLDER_NAME = "metadata"


class WaymoOpenDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        split_name: str,
        settings: Optional[DecoderSettings] = None,
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            split_name=split_name,
            settings=settings,
            **kwargs,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path) / split_name
        self.split_name = split_name
        dataset_name = f"Waymo Open Dataset - {split_name}"
        super().__init__(dataset_name=dataset_name, settings=settings, **kwargs)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return WaymoOpenDatasetSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return [f.name for f in self._dataset_path.iterdir()]

    def _decode_scene_names(self) -> List[SceneName]:
        return (
            []
        )  # TODO: Since one file can contain more that one scene we have to parse all files first to know how to split

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "waymo"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class WaymoOpenDatasetSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        record = self._dataset_path / scene_name
        frame_ids = list()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.append(frame_id)
        # with record.open("rb") as file:
        #     frame_ids = []
        #     while file:
        #         offset = file.tell()
        #         frame_ids.append(offset)
        #         header = file.read(12)
        #         if header == b"":
        #             break
        #         length, lengthcrc = struct.unpack("QI", header)
        #         file.seek(length + 4, 1)
        return set(frame_ids)

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return self.get_camera_names(scene_name=scene_name)

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        raise NotImplementedError()

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return decode_class_maps()

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[datetime]:
        return WaymoOpenDatasetCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[datetime]:
        raise ValueError("Directory decoder does not support lidar data!")

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[datetime]:
        return WaymoOpenDatasetFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        # frame_ids = sorted(self.get_frame_ids(scene_name=scene_name))
        record = self._dataset_path / scene_name
        frame_id_to_date_time_map = dict()
        for record, frame_id in get_record_iterator(record_path=record, read_frame=True):
            frame_id_to_date_time_map[frame_id] = datetime.fromtimestamp(record.timestamp_micros / 1000000)
        return frame_id_to_date_time_map

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[TDateTime]:
        raise ValueError("Loading from directory does not support radar data!")
