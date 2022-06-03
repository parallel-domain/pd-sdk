from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.directory.frame_decoder import DirectoryFrameDecoder
from paralleldomain.decoding.directory.sensor_decoder import DirectoryCameraSensorDecoder

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

IMAGE_FOLDER_NAME = "image"
SEMANTIC_SEGMENTATION_FOLDER_NAME = "semantic_segmentation"
METADATA_FOLDER_NAME = "metadata"


class DirectoryDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        class_map: ClassMap,
        settings: Optional[DecoderSettings] = None,
        image_folder: Optional[str] = IMAGE_FOLDER_NAME,
        semantic_segmentation_folder: Optional[str] = SEMANTIC_SEGMENTATION_FOLDER_NAME,
        metadata_folder: Optional[str] = METADATA_FOLDER_NAME,
        camera_name: Optional[str] = "default",
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            class_map=class_map,
            settings=settings,
            image_folder=image_folder,
            semantic_segmentation_folder=semantic_segmentation_folder,
            metadata_folder=metadata_folder,
            camera_name=camera_name,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path)

        self.class_map = class_map
        self.image_folder = image_folder
        self.semantic_segmentation_folder = semantic_segmentation_folder
        self.metadata_folder = metadata_folder
        self.camera_name = camera_name
        dataset_name = "-".join(list([dataset_path]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return DirectorySceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            class_map=self.class_map,
            settings=self.settings,
            image_folder=self.image_folder,
            semantic_segmentation_folder=self.semantic_segmentation_folder,
            metadata_folder=self.metadata_folder,
            camera_name=self.camera_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return [self.image_folder]

    def _decode_scene_names(self) -> List[SceneName]:
        return ()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "directory"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class DirectorySceneDecoder(SceneDecoder[None]):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        class_map: ClassMap,
        settings: DecoderSettings,
        image_folder: str,
        semantic_segmentation_folder: str,
        metadata_folder: str,
        camera_name: str,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._class_map = class_map
        self._image_folder = image_folder
        self._semantic_segmentation_folder = semantic_segmentation_folder
        self._metadata_folder = metadata_folder
        self._camera_name = camera_name

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        scene_images_folder = self._dataset_path / self._image_folder
        return {path.name for path in scene_images_folder.iterdir()}

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._camera_name]

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._camera_name]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        raise ValueError("Loading from directoy does not support lidar data!")

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return {AnnotationTypes.SemanticSegmentation2D: ClassMap(classes=self._class_map)}

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[None]:
        return DirectoryCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
            image_folder=self._image_folder,
            semantic_segmentation_folder=self._semantic_segmentation_folder,
            metadata_folder=self._metadata_folder,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[None]:
        raise ValueError("Directory decoder does not support lidar data!")

    def _create_frame_decoder(self, scene_name: SceneName, frame_id: FrameId, dataset_name: str) -> FrameDecoder[None]:
        return DirectoryFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            semantic_segmentation_folder=self._semantic_segmentation_folder,
            metadata_folder=self._metadata_folder,
            camera_name=self._camera_name,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, None]:
        return {fid: None for fid in self.get_frame_ids(scene_name=scene_name)}
