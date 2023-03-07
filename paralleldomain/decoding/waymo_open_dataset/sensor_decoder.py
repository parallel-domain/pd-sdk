from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.decoding.waymo_open_dataset.common import (
    get_cached_pre_calculated_scene_to_frame_info,
    get_record_iterator,
)
from paralleldomain.decoding.waymo_open_dataset.sensor_frame_decoder import (
    WaymoOpenDatasetCameraSensorFrameDecoder,
    WaymoOpenDatasetLidarSensorFrameDecoder,
)
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class WaymoOpenDatasetCameraSensorDecoder(CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.split_name = split_name
        self.use_precalculated_maps = use_precalculated_maps
        self._dataset_path = dataset_path

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        if self.use_precalculated_maps and self.split_name in ["training", "validation"]:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache, dataset_name=self.dataset_name, split_name=self.split_name
            )
            if self.scene_name in id_map:
                return [elem["frame_id"] for elem in id_map[self.scene_name]]

        record = self._dataset_path / self.scene_name
        frame_ids = list()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.append(frame_id)
        return frame_ids

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[None]:
        return CameraSensorFrame[None](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return WaymoOpenDatasetCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
        )


class WaymoOpenDatasetLidarSensorDecoder(LidarSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.split_name = split_name
        self.use_precalculated_maps = use_precalculated_maps
        self._dataset_path = dataset_path

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        if self.use_precalculated_maps and self.split_name in ["training", "validation"]:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache, dataset_name=self.dataset_name, split_name=self.split_name
            )
            if self.scene_name in id_map:
                return [elem["frame_id"] for elem in id_map[self.scene_name]]

        record = self._dataset_path / self.scene_name
        frame_ids = list()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.append(frame_id)
        return frame_ids

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, lidar_name: SensorName
    ) -> LidarSensorFrame[None]:
        return LidarSensorFrame[None](sensor_name=lidar_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[None]:
        return WaymoOpenDatasetLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
        )
