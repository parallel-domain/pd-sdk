import abc
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.common import resolve_scene_folder
from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class DirectoryCameraSensorDecoder(CameraSensorDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
        is_unordered_scene: bool,
        scene_decoder,
        img_file_extension: Optional[str] = "png",
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.dataset_path = dataset_path
        self.folder_to_data_type = folder_to_data_type
        self._metadata_folder = metadata_folder
        self._class_map = class_map
        self._img_file_extension = img_file_extension

    def _decode_frame_id_set(self) -> Set[FrameId]:
        default_folder = next(iter(self.folder_to_data_type.keys()))
        scene_images_folder = (
            resolve_scene_folder(dataset_path=self.dataset_path, scene_name=self.scene_name) / default_folder
        )
        return {path.stem for path in scene_images_folder.iterdir()}

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[None]:
        return DirectoryCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            img_file_extension=self._img_file_extension,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )


class DirectoryLidarSensorDecoder(LidarSensorDecoder[None], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.dataset_path = dataset_path
        self.folder_to_data_type = folder_to_data_type
        self._metadata_folder = metadata_folder
        self._class_map = class_map

    def _decode_frame_id_set(self) -> Set[FrameId]:
        default_folder = next(iter(self.folder_to_data_type.keys()))
        scene_images_folder = (
            resolve_scene_folder(dataset_path=self.dataset_path, scene_name=self.scene_name) / default_folder
        )
        return {path.name for path in scene_images_folder.iterdir()}

    @abc.abstractmethod
    def _create_lidar_sensor_frame_decoder(self, frame_id: FrameId) -> LidarSensorFrameDecoder[None]:
        pass
