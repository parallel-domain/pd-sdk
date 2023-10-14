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
        frame_id: FrameId,
        dataset_path: AnyPath,
        camera_names: List[SensorName],
        settings: DecoderSettings,
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
        self.camera_names = camera_names
        self.dataset_path = dataset_path

    def _decode_ego_pose(self) -> EgoPose:
        return EgoPose.from_transformation_matrix(np.eye(4))

    def _decode_available_sensor_names(self) -> List[SensorName]:
        return self.camera_names

    def _decode_available_camera_names(self) -> List[SensorName]:
        return self.camera_names

    def _decode_available_lidar_names(self) -> List[SensorName]:
        return list()

    def _decode_datetime(self) -> None:
        return None

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[None]:
        return CityscapesCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[None]:
        raise ValueError("Cityscapes does not contain lidar data!")

    def _decode_available_radar_names(self) -> List[SensorName]:
        """Not supported yet"""
        return list()

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[None]:
        raise ValueError("Cityscapes does not contain radar data!")

    def _decode_metadata(self) -> Dict[str, Any]:
        return dict()
