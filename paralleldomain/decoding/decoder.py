import abc
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar, Union

from paralleldomain import Scene
from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.decoding.common import LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import Dataset, DatasetMeta
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensor, LidarSensor
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.model.unordered_scene import UnorderedScene

T = TypeVar("T")

TDatasetType = TypeVar("TDatasetType", bound=Dataset)
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class DatasetDecoder(LazyLoadPropertyMixin, metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str, **kwargs):
        self.dataset_name = dataset_name
        self._scenes: Dict[SceneName, Scene] = dict()
        self._unordered_scenes: Dict[SceneName, UnorderedScene] = dict()

    def get_unique_id(
        self,
        scene_name: Optional[SceneName] = None,
        sensor_name: Optional[SensorName] = None,
        frame_id: Optional[FrameId] = None,
        extra: Optional[str] = None,
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    def get_unordered_scene_names(self) -> List[SceneName]:
        _unique_cache_key = self.get_unique_id(extra="unordered_scene_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=self._decode_unordered_scene_names,
        )

    def get_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        if scene_name in self.get_scene_names():
            return self.get_scene(scene_name=scene_name)

        return self.get_unordered_scene(scene_name=scene_name)

    def get_dataset_metadata(self) -> DatasetMeta:
        return self.lazy_load_cache.get_item(
            key=f"{self.dataset_name}-dataset_metadata",
            loader=self._decode_dataset_metadata,
        )

    def get_scene_names(self) -> List[SceneName]:
        _unique_cache_key = self.get_unique_id(extra="scene_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=self._decode_scene_names,
        )

    def get_scene(self, scene_name: SceneName) -> Scene:
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="scene")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_scene(scene_name=scene_name),
        )

    def _decode_scene(self, scene_name: SceneName) -> Scene:
        metadata = self.get_dataset_metadata()
        scene_decoder = self.create_scene_decoder(scene_name=scene_name)
        return Scene(
            name=scene_name, available_annotation_types=metadata.available_annotation_types, decoder=scene_decoder
        )

    def _decode_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        metadata = self.get_dataset_metadata()
        scene_decoder = self.create_scene_decoder(scene_name=scene_name)
        return UnorderedScene(
            name=scene_name, available_annotation_types=metadata.available_annotation_types, decoder=scene_decoder
        )

    @abc.abstractmethod
    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        pass

    @abc.abstractmethod
    def _decode_unordered_scene_names(self) -> List[SceneName]:
        pass

    @abc.abstractmethod
    def _decode_scene_names(self) -> List[SceneName]:
        pass

    @abc.abstractmethod
    def _decode_dataset_metadata(self) -> DatasetMeta:
        pass

    def get_dataset(self) -> Dataset:
        return Dataset(decoder=self)


class SceneDecoder(Generic[TDateTime], LazyLoadPropertyMixin, metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def get_unique_id(
        self,
        scene_name: Optional[SceneName] = None,
        sensor_name: Optional[SensorName] = None,
        frame_id: Optional[FrameId] = None,
        extra: Optional[str] = None,
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    # ------------------- Set Decoding Methods -------------------------------
    def get_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        # add caching here if needed
        return self._decode_set_metadata(scene_name=scene_name)

    def get_set_description(self, scene_name: SceneName) -> str:
        # add caching here if needed
        return self._decode_set_description(scene_name=scene_name)

    @abc.abstractmethod
    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def _decode_set_description(self, scene_name: SceneName) -> str:
        pass

    @abc.abstractmethod
    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        pass

    @abc.abstractmethod
    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_class_maps(self, scene_name: SceneName) -> Dict[str, ClassMap]:
        pass

    def get_sensor_names(self, scene_name: SceneName) -> List[str]:
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="sensor_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_names(scene_name=scene_name),
        )

    def get_camera_names(self, scene_name: SceneName) -> List[str]:
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="camera_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_camera_names(scene_name=scene_name),
        )

    def get_lidar_names(self, scene_name: SceneName) -> List[str]:
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="lidar_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_lidar_names(scene_name=scene_name),
        )

    def get_frame_ids(self, scene_name: SceneName) -> Set[FrameId]:
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="frame_ids")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_set(scene_name=scene_name),
        )

    def get_class_map(self, scene_name: SceneName, annotation_type: Type[T]) -> ClassMap:
        identifier = self._annotation_type_identifiers[annotation_type]
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="classmaps")
        class_maps = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_class_maps(scene_name=scene_name),
        )

        return class_maps[identifier]

    def _decode_lidar_sensor(
        self,
        scene_name: SceneName,
        sensor_name: SensorName,
        sensor_decoder: LidarSensorDecoder[TDateTime],
    ) -> LidarSensor[TDateTime]:
        return LidarSensor[TDateTime](sensor_name=sensor_name, decoder=sensor_decoder)

    def _decode_camera_sensor(
        self,
        scene_name: SceneName,
        sensor_name: SensorName,
        sensor_decoder: CameraSensorDecoder[TDateTime],
    ) -> CameraSensor[TDateTime]:
        return CameraSensor[TDateTime](sensor_name=sensor_name, decoder=sensor_decoder)

    def get_camera_sensor(self, scene_name: SceneName, sensor_name: SensorName) -> CameraSensor[TDateTime]:
        sensor_decoder = self._create_camera_sensor_decoder(
            scene_name=scene_name,
            sensor_name=sensor_name,
            dataset_name=self.dataset_name,
        )
        return self._decode_camera_sensor(scene_name=scene_name, sensor_name=sensor_name, sensor_decoder=sensor_decoder)

    def get_lidar_sensor(self, scene_name: SceneName, sensor_name: SensorName) -> LidarSensor[TDateTime]:
        sensor_decoder = self._create_lidar_sensor_decoder(
            scene_name=scene_name,
            sensor_name=sensor_name,
            dataset_name=self.dataset_name,
        )
        return self._decode_lidar_sensor(scene_name=scene_name, sensor_name=sensor_name, sensor_decoder=sensor_decoder)

    @abc.abstractmethod
    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, sensor_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, sensor_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[TDateTime]:
        pass

    @property
    def _annotation_type_identifiers(self) -> Dict[AnnotationType, AnnotationIdentifier]:
        available_annotation_types = {v: k for k, v in ANNOTATION_TYPE_MAP.items()}
        return available_annotation_types

    @abc.abstractmethod
    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[TDateTime]:
        pass

    def get_frame(
        self,
        scene_name: SceneName,
        frame_id: FrameId,
    ) -> Frame[TDateTime]:
        frame_decoder = self._create_frame_decoder(
            scene_name=scene_name,
            frame_id=frame_id,
            dataset_name=self.dataset_name,
        )
        return Frame(frame_id=frame_id, decoder=frame_decoder)

    @abc.abstractmethod
    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, TDateTime]:
        pass

    #

    def get_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, TDateTime]:
        _unique_cache_key = self.get_unique_id(scene_name=scene_name, extra="frame_id_to_date_time_map")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_to_date_time_map(scene_name=scene_name),
        )
