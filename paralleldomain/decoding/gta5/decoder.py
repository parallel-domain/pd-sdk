from typing import Dict, Optional, Union

from paralleldomain.decoding.gta5.common import GTA_CLASSES, IMAGE_FOLDER_NAME, SEMANTIC_SEGMENTATION_FOLDER_NAME
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import SceneDecoder
from paralleldomain.decoding.directory.decoder import DirectoryDatasetDecoder, DirectorySceneDecoder
from paralleldomain.decoding.gta5.frame_decoder import GTAFrameDecoder
from paralleldomain.decoding.gta5.sensor_decoder import GTACameraSensorDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class GTADatasetDecoder(DirectoryDatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        settings: Optional[DecoderSettings] = None,
        camera_name: Optional[str] = "default",
        **kwargs,
    ):
        super().__init__(
            dataset_path=dataset_path,
            settings=settings,
            class_map=None,
            image_folder=IMAGE_FOLDER_NAME,
            semantic_segmentation_folder=SEMANTIC_SEGMENTATION_FOLDER_NAME,
            metadata_folder=None,
            camera_name=camera_name,
        )
        self.dataset_name = "GTA5"

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return GTASceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            settings=self.settings,
            camera_name=self.camera_name,
        )

    @staticmethod
    def get_format() -> str:
        return "gta5"


class GTASceneDecoder(DirectorySceneDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        settings: DecoderSettings,
        camera_name: str,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            settings=settings,
            class_map=None,
            image_folder=IMAGE_FOLDER_NAME,
            semantic_segmentation_folder=SEMANTIC_SEGMENTATION_FOLDER_NAME,
            metadata_folder=None,
            camera_name=camera_name,
        )

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[None]:
        return GTACameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
        )

    def _create_frame_decoder(self, scene_name: SceneName, frame_id: FrameId, dataset_name: str) -> FrameDecoder[None]:
        return GTAFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            camera_name=self._camera_name,
        )

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return {AnnotationTypes.SemanticSegmentation2D: ClassMap(classes=GTA_CLASSES)}
