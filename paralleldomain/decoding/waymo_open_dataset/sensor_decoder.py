from datetime import datetime
from typing import Optional, Set

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
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class WaymoOpenDatasetCameraSensorDecoder(CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        is_unordered_scene: bool,
        index_folder: Optional[AnyPath],
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.split_name = split_name
        self._dataset_path = dataset_path
        self.index_folder = index_folder
        self.use_precalculated_maps = use_precalculated_maps
        if use_precalculated_maps is True and index_folder is None:
            raise ValueError("Index folder is required to use precalculated maps!")

    def _decode_frame_id_set(self) -> Set[FrameId]:
        if self.use_precalculated_maps and self.index_folder is not None:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                split_name=self.split_name,
                index_folder=self.index_folder,
            )
            if self.scene_name in id_map:
                return {elem["frame_id"] for elem in id_map[self.scene_name]}

        record = self._dataset_path / self.scene_name
        frame_ids = set()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.add(frame_id)
        return frame_ids

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[datetime]:
        return WaymoOpenDatasetCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self._dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
            index_folder=self.index_folder,
        )


class WaymoOpenDatasetLidarSensorDecoder(LidarSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        include_second_returns: bool,
        is_unordered_scene: bool,
        index_folder: Optional[AnyPath],
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.split_name = split_name
        self.include_second_returns = include_second_returns
        self.use_precalculated_maps = use_precalculated_maps
        self._dataset_path = dataset_path
        self.index_folder = index_folder
        if use_precalculated_maps is True and index_folder is None:
            raise ValueError("Index folder is required to use precalculated maps!")

    def _decode_frame_id_set(self) -> Set[FrameId]:
        if self.use_precalculated_maps and self.index_folder is not None:
            id_map = get_cached_pre_calculated_scene_to_frame_info(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                split_name=self.split_name,
                index_folder=self.index_folder,
            )
            if self.scene_name in id_map:
                return {elem["frame_id"] for elem in id_map[self.scene_name]}

        record = self._dataset_path / self.scene_name
        frame_ids = set()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.add(frame_id)
        return frame_ids

    def _create_lidar_sensor_frame_decoder(self, frame_id: FrameId) -> LidarSensorFrameDecoder[datetime]:
        return WaymoOpenDatasetLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
            include_second_returns=self.include_second_returns,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
            index_folder=self.index_folder,
        )
