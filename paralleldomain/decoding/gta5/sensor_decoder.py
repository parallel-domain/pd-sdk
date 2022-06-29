from functools import lru_cache
from typing import List, Optional

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.sensor_decoder import DirectoryCameraSensorDecoder
from paralleldomain.decoding.gta5.common import IMAGE_FOLDER_NAME, SEMANTIC_SEGMENTATION_FOLDER_NAME
from paralleldomain.decoding.gta5.sensor_frame_decoder import GTACameraSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath


class GTACameraSensorDecoder(DirectoryCameraSensorDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=dataset_path,
            settings=settings,
            image_folder=IMAGE_FOLDER_NAME,
            semantic_segmentation_folder=SEMANTIC_SEGMENTATION_FOLDER_NAME,
            metadata_folder=metadata_folder,
            class_map=class_map,
        )

    @lru_cache(maxsize=1)
    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return GTACameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            class_map=self._class_map,
            metadata_folder=self._metadata_folder,
        )
