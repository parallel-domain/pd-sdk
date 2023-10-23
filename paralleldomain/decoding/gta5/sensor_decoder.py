from functools import lru_cache
from typing import Dict, List, Optional

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.sensor_decoder import DirectoryCameraSensorDecoder
from paralleldomain.decoding.gta5.sensor_frame_decoder import GTACameraSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName
from paralleldomain.utilities.any_path import AnyPath


class GTACameraSensorDecoder(DirectoryCameraSensorDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: str,
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
            dataset_path=dataset_path,
            settings=settings,
            folder_to_data_type=folder_to_data_type,
            metadata_folder=metadata_folder,
            class_map=class_map,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )

    @lru_cache(maxsize=1)
    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[None]:
        return GTACameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
