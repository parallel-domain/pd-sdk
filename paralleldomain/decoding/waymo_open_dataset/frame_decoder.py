from datetime import datetime
from typing import Any, Dict, List, Optional

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder, TDateTime
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.decoding.waymo_open_dataset.common import (
    WAYMO_INDEX_TO_CAMERA_NAME,
    WAYMO_USE_ALL_LIDAR_NAME,
    WaymoFileAccessMixin,
)
from paralleldomain.decoding.waymo_open_dataset.sensor_frame_decoder import (
    WaymoOpenDatasetCameraSensorFrameDecoder,
    WaymoOpenDatasetLidarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class WaymoOpenDatasetFrameDecoder(FrameDecoder[datetime], WaymoFileAccessMixin):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        include_second_returns: bool,
        is_unordered_scene: bool,
        index_folder: Optional[AnyPath],
        scene_decoder,
    ):
        FrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        WaymoFileAccessMixin.__init__(self=self, record_path=dataset_path / scene_name)
        self.split_name = split_name
        self.include_second_returns = include_second_returns
        self.use_precalculated_maps = use_precalculated_maps
        self.dataset_path = dataset_path
        self.index_folder = index_folder
        if use_precalculated_maps is True and index_folder is None:
            raise ValueError("Index folder is required to use precalculated maps!")

    def _decode_ego_pose(self) -> EgoPose:
        raise ValueError("Loading from directory does not support ego pose!")

    def _decode_available_sensor_names(self) -> List[SensorName]:
        cam_names = self.get_camera_names()
        lidar_names = self.get_lidar_names()
        return cam_names + lidar_names

    def _decode_available_camera_names(self) -> List[SensorName]:
        record = self.get_record_at(frame_id=self.frame_id)
        return [WAYMO_INDEX_TO_CAMERA_NAME[img.name] for img in record.images]

    def _decode_available_lidar_names(self) -> List[SensorName]:
        return [WAYMO_USE_ALL_LIDAR_NAME]

    def _decode_datetime(self) -> datetime:
        record = self.get_record_at(frame_id=self.frame_id)
        return datetime.fromtimestamp(record.timestamp_micros / 1000000)

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[datetime]:
        return WaymoOpenDatasetCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            index_folder=self.index_folder,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[datetime]:
        return WaymoOpenDatasetLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
            include_second_returns=self.include_second_returns,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
            index_folder=self.index_folder,
        )

    def _decode_available_radar_names(
        self,
    ) -> List[SensorName]:
        raise ValueError("This dataset has no radar data!")

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[TDateTime]:
        raise ValueError("This dataset has no radar data!")

    def _decode_metadata(self) -> Dict[str, Any]:
        record = self.get_record_at(frame_id=self.frame_id)
        return dict(stats=record.context.stats, name=record.context.name)
