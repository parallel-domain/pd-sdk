from datetime import datetime
from typing import Any, Dict, List

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_things.common import LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME, frame_id_to_timestamp
from paralleldomain.decoding.flying_things.sensor_frame_decoder import FlyingThingsCameraSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class FlyingThingsFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        split_name: str,
        split_list: List[int],
        is_driving_subset: bool,
        is_full_dataset_format: bool = False,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._is_driving_subset = is_driving_subset
        self._is_full_dataset_format = is_full_dataset_format
        self._split_list = split_list
        self.dataset_path = dataset_path
        self._split_name = split_name

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        raise ValueError("FlyingThings does not support ego pose!")

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return [LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return [LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("FlyingThings does not support lidar data!")

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        return frame_id_to_timestamp(frame_id=frame_id)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return FlyingThingsCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            split_name=self._split_name,
            split_list=self._split_list,
            is_full_dataset_format=self._is_full_dataset_format,
            is_driving_subset=self._is_driving_subset,
        )

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[datetime]:
        raise ValueError("FlyingThings does not support lidar data!")

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("FlyingThings does not support radar data!")

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[datetime]:
        raise ValueError("FlyingThings does not support radar data!")

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()
