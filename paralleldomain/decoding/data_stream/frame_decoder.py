from datetime import datetime
from typing import Dict, Any, List

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream.data_accessor import DataStreamDataAccessor
from paralleldomain.decoding.data_stream.sensor_frame_decoder import DataStreamCameraSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    RadarSensorFrameDecoder,
    LidarSensorFrameDecoder,
    CameraSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import RadarSensorFrame, LidarSensorFrame, CameraSensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName, SceneName


class DataStreamFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self, dataset_name: str, scene_name: SceneName, settings: DecoderSettings, data_accessor: DataStreamDataAccessor
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._data_accessor = data_accessor

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        return self._data_accessor.get_ego_pose(frame_id=frame_id)

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return (
            self._decode_available_camera_names(frame_id=frame_id)
            + self._decode_available_lidar_names(frame_id=frame_id)
            + self._decode_available_radar_names(frame_id=frame_id)
        )

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        # TODO: afaik we don't support not rendering all sensors at all frames anyways?!
        return self._data_accessor.camera_names

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        return self._data_accessor.lidar_names

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        return self._data_accessor.radar_names

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        return self._data_accessor.get_frame_id_to_date_time_map()[frame_id]

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return DataStreamCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            settings=self.settings,
            data_accessor=self._data_accessor,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        # TODO: Can be removed after merge and rebase of other PR
        return CameraSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[datetime]:
        raise NotImplementedError("Lidar decoding not implemented")

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[datetime]:
        raise NotImplementedError("Radar decoding not implemented")

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        raise NotImplementedError("Lidar decoding not implemented")

    def _decode_radar_sensor_frame(
        self, decoder: RadarSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> RadarSensorFrame[datetime]:
        raise NotImplementedError("Radar decoding not implemented")
