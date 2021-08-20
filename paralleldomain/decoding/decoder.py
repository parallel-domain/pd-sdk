import abc
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

from paralleldomain import Scene
from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.decoding.common import create_cache_key
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import SensorDecoder
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import Dataset, DatasetMeta, SceneDataset
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import Sensor, SensorFrame
from paralleldomain.model.sensor_frame_set import SensorFrameSet
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorFrameSetName, SensorName
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE, LazyLoadCache, cache_max_ram_usage_factor

T = TypeVar("T")

TDatasetType = TypeVar("TDatasetType", bound=Dataset)
TSceneDatasetType = TypeVar("TSceneDatasetType", bound=SceneDataset)
TSensorFrameType = TypeVar("TSensorFrameType", bound=SensorFrame)


class Decoder(Generic[TDatasetType, TSensorFrameType], metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str, use_persistent_cache: bool = True):
        if use_persistent_cache:
            self._lazy_load_cache = LAZY_LOAD_CACHE
        else:
            self._lazy_load_cache = LazyLoadCache(max_ram_usage_factor=cache_max_ram_usage_factor)

        self.dataset_name = dataset_name
        self._scenes: Dict[SceneName, Scene] = dict()
        self._sensor_frame_sets: Dict[SensorFrameSetName, SensorFrameSet] = dict()

    def get_unique_id(
        self,
        set_name: Optional[SensorFrameSetName] = None,
        sensor_name: Optional[SensorName] = None,
        frame_id: Optional[FrameId] = None,
        extra: Optional[str] = None,
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name, set_name=set_name, sensor_name=sensor_name, frame_id=frame_id, extra=extra
        )

    @abc.abstractmethod
    def get_dataset(self) -> TDatasetType:
        pass

    # --------------------------------------------------------

    @abc.abstractmethod
    def _decode_sensor_frame_set_names(self) -> List[SensorFrameSetName]:
        pass

    @abc.abstractmethod
    def _decode_dataset_meta_data(self) -> DatasetMeta:
        pass

    def _decode_sensor_frame_set(self, set_name: SensorFrameSetName) -> SensorFrameSet:
        meta_data = self.get_dataset_meta_data()
        return SensorFrameSet(
            name=set_name, available_annotation_types=meta_data.available_annotation_types, decoder=self
        )

    def get_sensor_frame_set_names(self) -> List[SensorFrameSetName]:
        _unique_cache_key = self.get_unique_id(extra="sensor_frame_set_names")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=self._decode_sensor_frame_set_names,
        )

    def get_dataset_meta_data(self) -> DatasetMeta:
        return self._lazy_load_cache.get_item(
            key=f"{self.dataset_name}-dataset_meta_data",
            loader=self._decode_dataset_meta_data,
        )

    def get_sensor_frame_set(self, set_name: SensorFrameSetName) -> SensorFrameSet:
        _unique_cache_key = self.get_unique_id(set_name=set_name, extra="sensor_frame_set")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_frame_set(set_name=set_name),
        )

    # ------------------- Set Decoding Methods -------------------------------
    def get_set_metadata(self, set_name: SensorFrameSetName) -> Dict[str, Any]:
        # add caching here if needed
        return self._decode_set_metadata(set_name=set_name)

    def get_set_description(self, set_name: SensorFrameSetName) -> str:
        # add caching here if needed
        return self._decode_set_description(set_name=set_name)

    @abc.abstractmethod
    def _decode_set_metadata(self, set_name: SensorFrameSetName) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def _decode_set_description(self, set_name: SensorFrameSetName) -> str:
        pass

    @abc.abstractmethod
    def _decode_frame_id_set(self, set_name: SensorFrameSetName) -> Set[FrameId]:
        pass

    @abc.abstractmethod
    def _decode_sensor_names(self, set_name: SensorFrameSetName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_camera_names(self, set_name: SensorFrameSetName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_lidar_names(self, set_name: SensorFrameSetName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_class_maps(self, set_name: SensorFrameSetName) -> Dict[str, ClassMap]:
        pass

    def get_sensor_names(self, set_name: SensorFrameSetName) -> List[str]:
        _unique_cache_key = self.get_unique_id(set_name=set_name, extra="sensor_names")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_names(set_name=set_name),
        )

    def get_camera_names(self, set_name: SensorFrameSetName) -> List[str]:
        _unique_cache_key = self.get_unique_id(set_name=set_name, extra="camera_names")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_camera_names(set_name=set_name),
        )

    def get_lidar_names(self, set_name: SensorFrameSetName) -> List[str]:
        _unique_cache_key = self.get_unique_id(set_name=set_name, extra="lidar_names")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_lidar_names(set_name=set_name),
        )

    def get_frame_ids(self, set_name: SensorFrameSetName) -> Set[FrameId]:
        _unique_cache_key = self.get_unique_id(set_name=set_name, extra="frame_ids")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_set(set_name=set_name),
        )

    def get_class_map(self, set_name: SensorFrameSetName, annotation_type: Type[T]) -> ClassMap:
        identifier = self._annotation_type_identifiers[annotation_type]
        _unique_cache_key = self.get_unique_id(set_name=set_name, extra="classmaps")
        class_maps = self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_class_maps(set_name=set_name),
        )

        return class_maps[identifier]

    @abc.abstractmethod
    def _create_sensor_decoder(
        self, set_name: SensorFrameSetName, sensor_name: SensorName, dataset_name: str, lazy_load_cache: LazyLoadCache
    ) -> SensorDecoder:
        pass

    @abc.abstractmethod
    def _decode_sensor(
        self,
        set_name: SensorFrameSetName,
        sensor_name: SensorName,
        sensor_decoder: SensorDecoder,
    ) -> Sensor:
        pass

    def get_sensor(self, set_name: SensorFrameSetName, sensor_name: SensorName) -> Sensor[TSensorFrameType]:
        sensor_decoder = self._create_sensor_decoder(
            set_name=set_name,
            sensor_name=sensor_name,
            dataset_name=self.dataset_name,
            lazy_load_cache=self._lazy_load_cache,
        )
        return self._decode_sensor(set_name=set_name, sensor_name=sensor_name, sensor_decoder=sensor_decoder)

    @property
    def _annotation_type_identifiers(self) -> Dict[AnnotationType, AnnotationIdentifier]:
        available_annotation_types = {v: k for k, v in ANNOTATION_TYPE_MAP.items()}
        return available_annotation_types

    @abc.abstractmethod
    def _create_frame_decoder(
        self, set_name: SensorFrameSetName, frame_id: FrameId, dataset_name: str, lazy_load_cache: LazyLoadCache
    ) -> FrameDecoder:
        pass

    @abc.abstractmethod
    def _decode_frame(
        self, set_name: SensorFrameSetName, frame_id: FrameId, frame_decoder: FrameDecoder
    ) -> Frame[TSensorFrameType]:
        pass

    def get_frame(
        self,
        set_name: SensorFrameSetName,
        frame_id: FrameId,
    ) -> Frame:
        frame_decoder = self._create_frame_decoder(
            set_name=set_name, frame_id=frame_id, dataset_name=self.dataset_name, lazy_load_cache=self._lazy_load_cache
        )
        return self._decode_frame(set_name=set_name, frame_id=frame_id, frame_decoder=frame_decoder)


class TemporalDecoder(Decoder[TSceneDatasetType, TSensorFrameType], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _decode_scene_names(self) -> List[SceneName]:
        pass

    def _decode_scene(self, scene_name: SceneName) -> Scene:
        meta_data = self.get_dataset_meta_data()
        return Scene(name=scene_name, available_annotation_types=meta_data.available_annotation_types, decoder=self)

    def get_scene(self, scene_name: SceneName) -> Scene:
        _unique_cache_key = self.get_unique_id(set_name=scene_name, extra="scene")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_scene(scene_name=scene_name),
        )

    def get_scene_names(self) -> List[SceneName]:
        _unique_cache_key = self.get_unique_id(extra="scene_names")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=self._decode_scene_names,
        )

    def get_sensor_frame_set(self, set_name: SensorFrameSetName) -> SensorFrameSet:
        if set_name in self.get_scene_names():
            return self.get_scene(scene_name=set_name)

        return super().get_sensor_frame_set(set_name=set_name)

    # ------------------- Set Decoding Methods -------------------------------

    @abc.abstractmethod
    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        pass

    def get_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        _unique_cache_key = self.get_unique_id(set_name=scene_name, extra="frame_id_to_date_time_map")
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_to_date_time_map(scene_name=scene_name),
        )
