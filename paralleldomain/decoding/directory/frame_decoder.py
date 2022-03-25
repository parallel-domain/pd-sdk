from typing import List

import numpy as np

from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryCameraSensorFrameDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class DirectoryFrameDecoder(FrameDecoder[None]):
    def __init__(
        self, dataset_name: str, scene_name: SceneName, dataset_path: AnyPath, settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        raise ValueError("Loading from directoy does not support ego pose!")

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return ["default"]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return ["default"]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("Loading from directoy does not support lidar data!")

    def _decode_datetime(self, frame_id: FrameId) -> None:
        return None

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return DirectoryCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[None], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[None]:
        return CameraSensorFrame[None](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[None]:
        raise ValueError("Loading from directoy does not support lidar data!")

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[None], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[None]:
        raise ValueError("Loading from directoy does not support lidar data!")
