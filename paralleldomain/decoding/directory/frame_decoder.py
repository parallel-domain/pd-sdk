from typing import List

import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryCameraSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder, TDateTime
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class DirectoryFrameDecoder(FrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        semantic_segmentation_folder: str,
        metadata_folder: str,
        camera_name: str,
        class_map: List[ClassDetail],
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path
        self.image_folder = image_folder
        self.semantic_segmentation_folder = semantic_segmentation_folder
        self.camera_name = camera_name
        self.metadata_folder = metadata_folder
        self._class_map = class_map

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        raise ValueError("Loading from directory does not support ego pose!")

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return [self.camera_name]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return [self.camera_name]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("Loading from directory does not support lidar data!")

    def _decode_datetime(self, frame_id: FrameId) -> None:
        return None

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return DirectoryCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            image_folder=self.image_folder,
            semantic_segmentation_folder=self.semantic_segmentation_folder,
            metadata_folder=self.metadata_folder,
            class_map=self._class_map,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[None], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[None]:
        return CameraSensorFrame[None](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[None]:
        raise ValueError("Loading from directory does not support lidar data!")

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[None], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[None]:
        raise ValueError("Loading from directoy does not support lidar data!")

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("Loading from directory does not support radar data!")

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[TDateTime]:
        raise ValueError("Loading from directory does not support radar data!")

    def _decode_radar_sensor_frame(
        self, decoder: RadarSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> RadarSensorFrame[TDateTime]:
        raise ValueError("Loading from directory does not support radar data!")
