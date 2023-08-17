from typing import Any, Dict, List, Optional, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.directory.common import decode_class_maps, resolve_scene_folder
from paralleldomain.decoding.directory.frame_decoder import DirectoryFrameDecoder
from paralleldomain.decoding.directory.sensor_decoder import DirectoryCameraSensorDecoder, DirectoryLidarSensorDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationTypes, AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.radar_point_cloud import RadarPointCloud
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

AVAILABLE_ANNOTATION_IDENTIFIERS = [
    AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D),
]
METADATA_FOLDER_NAME = "metadata"


class DirectoryDatasetDecoder(DatasetDecoder):
    """
    the DirectoryDatasetDecoder assumes:
    - only one sensor
    - one folder per datatype (images, point_cloud, annotations, etc.)
    - matching filenames between folders to link frame data
    """

    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        class_map: List[ClassDetail],
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        settings: Optional[DecoderSettings] = None,
        metadata_folder: Optional[str] = METADATA_FOLDER_NAME,
        sensor_name: Optional[str] = "default",
        **kwargs,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            class_map=class_map,
            settings=settings,
            folder_to_data_type=folder_to_data_type,
            metadata_folder=metadata_folder,
            sensor_name=sensor_name,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path)

        self.class_map = [ClassDetail(**c) if isinstance(c, Dict) else c for c in class_map]
        self.folder_to_data_type = folder_to_data_type
        self.metadata_folder = metadata_folder
        self.sensor_name = sensor_name
        dataset_name = "-".join(list([str(dataset_path)]))
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        return DirectorySceneDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            class_map=self.class_map,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self.metadata_folder,
            sensor_name=self.sensor_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return [folder.name for folder in self._dataset_path.iterdir() if folder.is_dir()]

    def _decode_scene_names(self) -> List[SceneName]:
        return list()

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=AVAILABLE_ANNOTATION_IDENTIFIERS,
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
        class_map: List[ClassDetail],
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        metadata_folder: Optional[str],
        sensor_name: Optional[str] = "default",
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._class_map = class_map
        self._folder_to_data_type = folder_to_data_type
        self._metadata_folder = metadata_folder
        self._sensor_name = sensor_name

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        default_folder = next(iter(self._folder_to_data_type.keys()))
        scene_images_folder = (
            resolve_scene_folder(dataset_path=self._dataset_path, scene_name=scene_name) / default_folder
        )
        return {path.stem for path in scene_images_folder.iterdir()}

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._sensor_name]

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._sensor_name] if any([d == Image for d in self._folder_to_data_type.values()]) else []

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._sensor_name] if any([d == PointCloud for d in self._folder_to_data_type.values()]) else []

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        return [self._sensor_name] if any([d == RadarPointCloud for d in self._folder_to_data_type.values()]) else []

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationIdentifier, ClassMap]:
        annotation_types = [
            annotation_type
            for annotation_type in self._folder_to_data_type.values()
            if isinstance(annotation_type, AnnotationIdentifier)
        ]
        return decode_class_maps(class_map=self._class_map, annotation_types=annotation_types)

    def _decode_available_annotation_identifiers(self, scene_name: SceneName) -> List[AnnotationIdentifier]:
        return AVAILABLE_ANNOTATION_IDENTIFIERS

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[None]:
        return DirectoryCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=scene_name,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[None]:
        return DirectoryLidarSensorDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
        )

    def _create_frame_decoder(self, scene_name: SceneName, frame_id: FrameId, dataset_name: str) -> FrameDecoder[None]:
        return DirectoryFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            sensor_name=self._sensor_name,
            class_map=self._class_map,
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, None]:
        return {fid: None for fid in self.get_frame_ids(scene_name=scene_name)}

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[TDateTime]:
        raise ValueError("Loading from directory does not support radar data!")
