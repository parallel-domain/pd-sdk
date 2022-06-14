from datetime import datetime
from functools import lru_cache
from typing import List, Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class DirectoryCameraSensorDecoder(CameraSensorDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        semantic_segmentation_folder: str,
        metadata_folder: str,
        class_map: List[ClassDetail],
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path
        self.image_folder = image_folder
        self.semantic_segmentation_folder = semantic_segmentation_folder
        self.metadata_folder = metadata_folder
        self._class_map = class_map
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        scene_images_folder = self.dataset_path / self.image_folder
        return {path.name for path in scene_images_folder.iterdir()}

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
            image_folder=self.image_folder,
            semantic_segmentation_folder=self.semantic_segmentation_folder,
            metadata_folder=self.metadata_folder,
            class_map=self._class_map,
        )
