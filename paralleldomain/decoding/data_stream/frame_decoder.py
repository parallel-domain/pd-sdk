from datetime import datetime
from typing import Any, Dict, List

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream.data_accessor import DataStreamDataAccessor
from paralleldomain.decoding.data_stream.sensor_frame_decoder import DataStreamCameraSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class DataStreamFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        frame_id: FrameId,
        settings: DecoderSettings,
        data_accessor: DataStreamDataAccessor,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self._data_accessor = data_accessor

    def _decode_ego_pose(self) -> EgoPose:
        return self._data_accessor.get_ego_pose(frame_id=self.frame_id)

    def _decode_available_sensor_names(self) -> List[SensorName]:
        return (
            self._decode_available_camera_names()
            + self._decode_available_lidar_names()
            + self._decode_available_radar_names()
        )

    def _decode_available_camera_names(self) -> List[SensorName]:
        return list(self._data_accessor.cameras[self.frame_id].keys())

    def _decode_available_lidar_names(self) -> List[SensorName]:
        return list(self._data_accessor.lidars[self.frame_id].keys())

    def _decode_available_radar_names(self) -> List[SensorName]:
        return list(self._data_accessor.radars[self.frame_id].keys())

    def _decode_metadata(self) -> Dict[str, Any]:
        return dict()

    def _decode_datetime(self) -> datetime:
        return self._data_accessor.get_frame_id_to_date_time_map()[self.frame_id]

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[datetime]:
        return DataStreamCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            settings=self.settings,
            data_accessor=self._data_accessor,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[datetime]:
        raise NotImplementedError("Lidar decoding not implemented")

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[datetime]:
        raise NotImplementedError("Radar decoding not implemented")

    def _decode_lidar_sensor_frame(self, decoder: LidarSensorFrameDecoder[datetime]) -> LidarSensorFrame[datetime]:
        raise NotImplementedError("Lidar decoding not implemented")

    def _decode_radar_sensor_frame(self, decoder: RadarSensorFrameDecoder[datetime]) -> RadarSensorFrame[datetime]:
        raise NotImplementedError("Radar decoding not implemented")
