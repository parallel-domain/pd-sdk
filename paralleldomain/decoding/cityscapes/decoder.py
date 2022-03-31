from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.cityscapes.common import CITYSCAPE_CLASSES, get_scene_path
from paralleldomain.decoding.cityscapes.frame_decoder import CityscapesFrameDecoder
from paralleldomain.decoding.cityscapes.sensor_decoder import CityscapesCameraSensorDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

IMAGE_FOLDER_NAME = "leftImg8bit"


class CityscapesDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        splits: Optional[List[str]] = None,
        settings: Optional[DecoderSettings] = None,
        **kwargs,
    ):
        self._init_kwargs = dict(dataset_path=dataset_path, settings=settings, splits=splits)
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        if splits is None:
            splits = ["test", "train", "val"]
        self.splits = splits
        dataset_name = "-".join(list(["cityscapes"] + splits))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return CityscapesSceneDecoder(
            dataset_path=self._dataset_path, dataset_name=self.dataset_name, settings=self.settings
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        scene_names = list()
        for split_name in self.splits:
            split_scenes_folder = self._dataset_path / IMAGE_FOLDER_NAME / split_name
            for folder_path in split_scenes_folder.iterdir():
                scene_name = f"{split_name}-{folder_path.name}"
                scene_names.append(scene_name)
        return scene_names

    def _decode_scene_names(self) -> List[SceneName]:
        return list()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_types=[AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D],
            custom_attributes=dict(splits=self.splits),
        )

    @staticmethod
    def get_format() -> str:
        return "cityscapes"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class CityscapesSceneDecoder(SceneDecoder[None]):
    def __init__(self, dataset_path: Union[str, AnyPath], dataset_name: str, settings: DecoderSettings):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._camera_names = [IMAGE_FOLDER_NAME]

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        frame_ids = set()
        for camera in self._camera_names:
            scene_images_folder = get_scene_path(
                dataset_path=self._dataset_path, scene_name=scene_name, camera_name=camera
            )
            file_names = [path.name for path in scene_images_folder.iterdir()]
            frame_ids.update(file_names)
        return frame_ids

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._camera_names

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._camera_names

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return list()

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return {AnnotationTypes.SemanticSegmentation2D: ClassMap(classes=CITYSCAPE_CLASSES)}

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[None]:
        return CityscapesCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[None]:
        raise ValueError("Cityscapes does not contain lidar data!")

    def _create_frame_decoder(self, scene_name: SceneName, frame_id: FrameId, dataset_name: str) -> FrameDecoder[None]:
        return CityscapesFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            camera_names=self._camera_names,
            settings=self.settings,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, None]:
        return {fid: None for fid in self.get_frame_ids(scene_name=scene_name)}
