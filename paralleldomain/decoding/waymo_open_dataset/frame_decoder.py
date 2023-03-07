from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder, TDateTime
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.decoding.waymo_open_dataset.common import (
    WAYMO_INDEX_TO_CAMERA_NAME,
    WAYMO_INDEX_TO_LIDAR_NAME,
    WaymoFileAccessMixin,
)
from paralleldomain.decoding.waymo_open_dataset.sensor_frame_decoder import WaymoOpenDatasetCameraSensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class WaymoOpenDatasetFrameDecoder(FrameDecoder[datetime], WaymoFileAccessMixin):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
    ):
        FrameDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        WaymoFileAccessMixin.__init__(self=self, record_path=dataset_path / scene_name)
        self.split_name = split_name
        self.use_precalculated_maps = use_precalculated_maps
        self.dataset_path = dataset_path

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        raise ValueError("Loading from directory does not support ego pose!")

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return self.get_camera_names(frame_id=frame_id)

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        record = self.get_record_at(frame_id=frame_id)
        return [WAYMO_INDEX_TO_CAMERA_NAME[img.name] for img in record.images]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        record = self.get_record_at(frame_id=frame_id)
        return [WAYMO_INDEX_TO_LIDAR_NAME[laser.name] for laser in record.lasers]

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        record = self.get_record_at(frame_id=frame_id)
        return datetime.fromtimestamp(record.timestamp_micros / 1000000)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return WaymoOpenDatasetCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            use_precalculated_maps=self.use_precalculated_maps,
            split_name=self.split_name,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[datetime]:
        raise NotImplementedError()

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        raise NotImplementedError()

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("This dataset has no radar data!")

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[TDateTime]:
        raise ValueError("his dataset has no radar data!")

    def _decode_radar_sensor_frame(
        self, decoder: RadarSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> RadarSensorFrame[TDateTime]:
        raise ValueError("his dataset has no radar data!")

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        record = self.get_record_at(frame_id=frame_id)
        return dict(stats=record.context.stats, name=record.context.name)
