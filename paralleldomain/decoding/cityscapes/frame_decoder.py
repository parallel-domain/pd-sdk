from typing import Any, Dict, List

import numpy as np

from paralleldomain.decoding.cityscapes.sensor_frame_decoder import CityscapesCameraSensorFrameDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class CityscapesFrameDecoder(FrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        camera_names: List[SensorName],
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.camera_names = camera_names
        self.dataset_path = dataset_path

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        return EgoPose.from_transformation_matrix(np.eye(4))

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return self.camera_names

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return self.camera_names

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        return list()

    def _decode_datetime(self, frame_id: FrameId) -> None:
        return None

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return CityscapesCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
        )

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[None]:
        raise ValueError("Cityscapes does not contain lidar data!")

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        """Not supported yet"""
        return list()

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[None]:
        raise ValueError("Cityscapes does not contain radar data!")

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()
