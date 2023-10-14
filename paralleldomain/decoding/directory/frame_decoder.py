from typing import Any, Dict, List, Optional

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
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.radar_point_cloud import RadarPointCloud
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_json


class DirectoryFrameDecoder(FrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        metadata_folder: Optional[str],
        sensor_name: str,
        class_map: List[ClassDetail],
        is_unordered_scene: bool,
        scene_decoder,
        img_file_extension: Optional[str] = "png",
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.dataset_path = dataset_path
        self._folder_to_data_type = folder_to_data_type
        self._sensor_name = sensor_name
        self._metadata_folder = metadata_folder
        self._class_map = class_map
        self._img_file_extension = img_file_extension

    def _decode_ego_pose(self) -> EgoPose:
        raise ValueError("Loading from directory does not support ego pose!")

    def _decode_available_sensor_names(self) -> List[SensorName]:
        return [self._sensor_name]

    def _decode_available_camera_names(self) -> List[SensorName]:
        return [self._sensor_name] if any([d == Image for d in self._folder_to_data_type.values()]) else []

    def _decode_available_lidar_names(self) -> List[SensorName]:
        return [self._sensor_name] if any([d == PointCloud for d in self._folder_to_data_type.values()]) else []

    def _decode_available_radar_names(self) -> List[SensorName]:
        return [self._sensor_name] if any([d == RadarPointCloud for d in self._folder_to_data_type.values()]) else []

    def _decode_datetime(self) -> None:
        return None

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[None]:
        return DirectoryCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            img_file_extension=self._img_file_extension,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[None]:
        raise ValueError("Loading from directory does not support lidar data!")

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[TDateTime]:
        raise ValueError("Loading from directory does not support radar data!")

    def _decode_metadata(self) -> Dict[str, Any]:
        if self._metadata_folder is None:
            return dict()
        metadata_path = self.dataset_path / self._metadata_folder / f"{AnyPath(self.frame_id).stem + '.json'}"
        return read_json(metadata_path)
