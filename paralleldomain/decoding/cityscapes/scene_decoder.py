from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

from paralleldomain.decoding.cityscapes.common import decode_class_maps, get_scene_path
from paralleldomain.decoding.cityscapes.frame_decoder import CityscapesFrameDecoder
from paralleldomain.decoding.cityscapes.sensor_decoder import CityscapesCameraSensorDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import SceneDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.scene_access_decoder import SceneAccessDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

_AVAILABLE_ANNOTATION_TYPES = [AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D]
_AVAILABLE_ANNOTATION_IDENTIFIERS = [AnnotationIdentifier(annotation_type=t) for t in _AVAILABLE_ANNOTATION_TYPES]
IMAGE_FOLDER_NAME = "leftImg8bit"


T = TypeVar("T")

TDatasetType = TypeVar("TDatasetType", bound=Dataset)
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class CityscapesSceneDecoder(SceneDecoder[None]):
    def __init__(
        self, dataset_path: Union[str, AnyPath], dataset_name: str, scene_name: SceneName, settings: DecoderSettings
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)

        super().__init__(dataset_name=dataset_name, settings=settings, scene_name=scene_name)
        self._camera_names = [IMAGE_FOLDER_NAME]

    def _decode_set_metadata(self) -> Dict[str, Any]:
        return dict()

    def _decode_set_description(self) -> str:
        return ""

    def _decode_radar_names(self) -> List[SensorName]:
        """Radar not supported atm"""
        return list()

    def _create_radar_sensor_decoder(self, sensor_name: SensorName) -> RadarSensorDecoder[None]:
        raise ValueError("Cityscapes does not contain radar data!")

    def _decode_frame_id_set(self) -> Set[FrameId]:
        frame_ids = set()
        for camera in self._camera_names:
            scene_images_folder = get_scene_path(
                dataset_path=self._dataset_path, scene_name=self.scene_name, camera_name=camera
            )
            file_names = [path.name for path in scene_images_folder.iterdir()]
            frame_ids.update(file_names)
        return frame_ids

    def _decode_sensor_names(self) -> List[SensorName]:
        return self._camera_names

    def _decode_camera_names(self) -> List[SensorName]:
        return self._camera_names

    def _decode_lidar_names(self) -> List[SensorName]:
        return list()

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return decode_class_maps()

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return _AVAILABLE_ANNOTATION_IDENTIFIERS

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[None]:
        return CityscapesCameraSensorDecoder(
            dataset_name=self.dataset_name,
            dataset_path=self._dataset_path,
            scene_name=self.scene_name,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=True,
            sensor_name=sensor_name,
        )

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[None]:
        raise ValueError("Cityscapes does not contain lidar data!")

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[None]:
        return CityscapesFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            camera_names=self._camera_names,
            frame_id=frame_id,
            settings=self.settings,
            scene_decoder=self,
            is_unordered_scene=True,
        )

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, None]:
        return {fid: None for fid in self.get_frame_ids()}
