import abc
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Set, Dict

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
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path
        self.folder_to_data_type = folder_to_data_type
        self._metadata_folder = metadata_folder
        self._class_map = class_map
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        default_folder = next(iter(self.folder_to_data_type.keys()))
        scene_images_folder = (
            resolve_scene_folder(dataset_path=self.dataset_path, scene_name=self.scene_name) / default_folder
        )
        return {path.stem for path in scene_images_folder.iterdir()}

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[None]:
        return CameraSensorFrame[None](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return DirectoryCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
        )


class DirectoryLidarSensorDecoder(LidarSensorDecoder[None], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path
        self.folder_to_data_type = folder_to_data_type
        self._metadata_folder = metadata_folder
        self._class_map = class_map
        self._create_lidar_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_lidar_sensor_frame_decoder)

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        default_folder = next(iter(self.folder_to_data_type.keys()))
        scene_images_folder = (
            resolve_scene_folder(dataset_path=self.dataset_path, scene_name=self.scene_name) / default_folder
        )
        return {path.name for path in scene_images_folder.iterdir()}

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> LidarSensorFrame[None]:
        return LidarSensorFrame[None](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self):
        # todo: implement a DirectoryLidarSensorFrameDecoder
        pass
