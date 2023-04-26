import json
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.decoding.waymo_open_dataset.common import (
    WAYMO_INDEX_TO_CAMERA_NAME,
    WAYMO_USE_ALL_LIDAR_NAME,
    decode_class_maps,
    get_cached_pre_calculated_scene_to_frame_info,
    get_record_iterator,
)
from paralleldomain.decoding.waymo_open_dataset.frame_decoder import WaymoOpenDatasetFrameDecoder
from paralleldomain.decoding.waymo_open_dataset.sensor_decoder import (
    WaymoOpenDatasetCameraSensorDecoder,
    WaymoOpenDatasetLidarSensorDecoder,
)
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
        use_precalculated_maps: bool = True,
        include_second_returns: bool = True,
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
        self.include_second_returns = include_second_returns
        self.use_precalculated_maps = use_precalculated_maps
        dataset_name = f"Waymo Open Dataset - {split_name}"
        super().__init__(dataset_name=dataset_name, settings=settings, **kwargs)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return WaymoOpenDatasetSceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            split_name=self.split_name,
            use_precalculated_maps=self.use_precalculated_maps,
            include_second_returns=self.include_second_returns,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return self.get_scene_names()

    def _decode_scene_names(self) -> List[SceneName]:
        if self.use_precalculated_maps and self.split_name in ["training", "validation"]:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache, dataset_name=self.dataset_name, split_name=self.split_name
            )
            return sorted(list(id_map.keys()))

        return sorted([f.name for f in self._dataset_path.iterdir()])
        # return []

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[
                AnnotationTypes.SemanticSegmentation2D,
                AnnotationTypes.InstanceSegmentation2D,
                AnnotationTypes.BoundingBoxes3D,
            ],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "waymo_open_dataset"

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
        use_precalculated_maps: bool,
        split_name: str,
        include_second_returns: bool,
    ):
        self.split_name = split_name
        self.include_second_returns = include_second_returns
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        self.use_precalculated_maps = use_precalculated_maps
        super().__init__(dataset_name=dataset_name, settings=settings)

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        if self.use_precalculated_maps and self.split_name in ["training", "validation"]:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache, dataset_name=self.dataset_name, split_name=self.split_name
            )
            if scene_name in id_map:
                return {elem["frame_id"] for elem in id_map[scene_name]}
        record = self._dataset_path / scene_name
        frame_ids = list()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.append(frame_id)
        return set(frame_ids)

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        cam_names = self.get_camera_names(scene_name=scene_name)
        lidar_names = self.get_lidar_names(scene_name=scene_name)
        return cam_names + lidar_names

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return list(WAYMO_INDEX_TO_CAMERA_NAME.values())

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return [WAYMO_USE_ALL_LIDAR_NAME]

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
            split_name=self.split_name,
            use_precalculated_maps=self.use_precalculated_maps,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[datetime]:
        return WaymoOpenDatasetLidarSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
            split_name=self.split_name,
            use_precalculated_maps=self.use_precalculated_maps,
            include_second_returns=self.include_second_returns,
        )

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[datetime]:
        return WaymoOpenDatasetFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
            include_second_returns=self.include_second_returns,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        frame_id_to_date_time_map = dict()
        if self.use_precalculated_maps and self.split_name in ["training", "validation"]:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache, dataset_name=self.dataset_name, split_name=self.split_name
            )
            for elem in id_map[scene_name]:
                frame_id_to_date_time_map[elem["frame_id"]] = datetime.fromtimestamp(elem["timestamp_micros"] / 1000000)
        else:
            record = self._dataset_path / scene_name
            for record, frame_id in get_record_iterator(record_path=record, read_frame=True):
                frame_id_to_date_time_map[frame_id] = datetime.fromtimestamp(record.timestamp_micros / 1000000)
        return frame_id_to_date_time_map

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        """Radar not supported"""
        return list()

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[TDateTime]:
        raise ValueError("This dataset has no radar data!")
