from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.frame_decoder import DirectoryFrameDecoder
from paralleldomain.decoding.gta5.common import IMAGE_FOLDER_NAME, SEMANTIC_SEGMENTATION_FOLDER_NAME
from paralleldomain.decoding.gta5.sensor_frame_decoder import GTACameraSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder

from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath


class GTAFrameDecoder(DirectoryFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        camera_name: str,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=dataset_path,
            settings=settings,
            image_folder=IMAGE_FOLDER_NAME,
            semantic_segmentation_folder=SEMANTIC_SEGMENTATION_FOLDER_NAME,
            metadata_folder=None,
            camera_name=camera_name,
        )

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return GTACameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
        )
