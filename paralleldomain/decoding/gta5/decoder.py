from typing import Dict, List, Optional, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import SceneDecoder
from paralleldomain.decoding.directory.decoder import DirectoryDatasetDecoder, DirectorySceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.gta5.common import GTA_CLASSES
from paralleldomain.decoding.gta5.frame_decoder import GTAFrameDecoder
from paralleldomain.decoding.gta5.sensor_decoder import GTACameraSensorDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

FOLDER_TO_DATA_TYPE = {
    "images": Image,
    "labels": AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
}


class GTADatasetDecoder(DirectoryDatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        class_map: Optional[List[ClassDetail]] = None,
        folder_to_data_type: Optional[Dict[str, SensorDataCopyTypes]] = None,
        settings: Optional[DecoderSettings] = None,
        metadata_folder: Optional[str] = None,
        sensor_name: Optional[str] = "default",
        **kwargs,
    ):
        if class_map is None:
            class_map = GTA_CLASSES
        class_map = GTA_CLASSES if class_map is None else class_map
        folder_to_data_type = FOLDER_TO_DATA_TYPE if folder_to_data_type is None else folder_to_data_type

        super().__init__(
            dataset_path=dataset_path,
            class_map=class_map,
            settings=settings,
            folder_to_data_type=folder_to_data_type,
            metadata_folder=metadata_folder,
            sensor_name=sensor_name,
            **kwargs,
        )
        self.dataset_name = "GTA5"

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        # todo: change?
        return ["default_scene"]

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return GTASceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            class_map=self.class_map,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self.metadata_folder,
            sensor_name=self.sensor_name,
            scene_name=scene_name,
        )

    @staticmethod
    def get_format() -> str:
        return "gta5"


class GTASceneDecoder(DirectorySceneDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        scene_name: SceneName,
        class_map: List[ClassDetail],
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        metadata_folder: Optional[str],
        sensor_name: Optional[str] = "default",
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            class_map=class_map,
            settings=settings,
            folder_to_data_type=folder_to_data_type,
            metadata_folder=metadata_folder,
            sensor_name=sensor_name,
            scene_name=scene_name,
        )

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[None]:
        return GTACameraSensorDecoder(
            dataset_name=self.dataset_name,
            sensor_name=sensor_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            scene_decoder=self,
            is_unordered_scene=True,
        )

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[None]:
        return GTAFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=frame_id,
            dataset_path=self._dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            sensor_name=self._sensor_name,
            class_map=self._class_map,
            scene_decoder=self,
            is_unordered_scene=True,
        )
