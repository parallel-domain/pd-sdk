import abc
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Union

from paralleldomain import Scene
from paralleldomain.decoding.common import DecoderSettings, LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import Dataset, DatasetMeta
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensor, LidarSensor, RadarSensor
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath

T = TypeVar("T")

TDatasetType = TypeVar("TDatasetType", bound=Dataset)
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class DatasetDecoder(LazyLoadPropertyMixin, metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str, settings: Optional[DecoderSettings], **kwargs):
        if settings is None:
            settings = DecoderSettings()
        self.settings = settings
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

    def get_scene(self, scene_name: SceneName) -> Scene:
        _unique_cache_key = create_cache_key(scene_name=scene_name, extra="scene", dataset_name=self.dataset_name)
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_scene(scene_name=scene_name),
        )

    def get_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        if scene_name in self.get_scene_names():
            return self.get_scene(scene_name=scene_name)

        return self._decode_unordered_scene(scene_name=scene_name)

    def _decode_scene(self, scene_name: SceneName) -> Scene:
        scene_decoder = self.create_scene_decoder(scene_name=scene_name)
        scene = Scene(decoder=scene_decoder)
        if self.settings.model_decorator is not None:
            scene = self.settings.model_decorator(scene)
        return scene

    def _decode_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        scene_decoder = self.create_scene_decoder(scene_name=scene_name)
        scene = UnorderedScene(decoder=scene_decoder)
        if self.settings.model_decorator is not None:
            scene = self.settings.model_decorator(scene)
        return scene

    @abc.abstractmethod
    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        pass

    def get_unordered_scene_names(self) -> List[SceneName]:
        _unique_cache_key = self.get_unique_id(extra="unordered_scene_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=self._decode_unordered_scene_names,
        )

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
        dataset = Dataset(decoder=self)
        if self.settings.model_decorator is not None:
            dataset = self.settings.model_decorator(dataset)
        return dataset

    @staticmethod
    @abc.abstractmethod
    def get_format() -> str:
        pass

    @abc.abstractmethod
    def get_path(self) -> Optional[AnyPath]:
        pass

    @abc.abstractmethod
    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        pass


class SceneDecoder(Generic[TDateTime], LazyLoadPropertyMixin, metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str, scene_name: SceneName, settings: DecoderSettings):
        self.settings = settings
        self.dataset_name = dataset_name
        self.scene_name = scene_name

    def get_unique_id(
        self,
        sensor_name: Optional[SensorName] = None,
        frame_id: Optional[FrameId] = None,
        extra: Optional[str] = None,
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    def get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        _unique_cache_key = self.get_unique_id(extra="available_annotations")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_annotation_identifiers(),
        )

    # ------------------- Set Decoding Methods -------------------------------
    def get_set_metadata(self) -> Dict[str, Any]:
        # add caching here if needed
        return self._decode_set_metadata()

    def get_set_description(self) -> str:
        # add caching here if needed
        return self._decode_set_description()

    @abc.abstractmethod
    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        pass

    @abc.abstractmethod
    def _decode_set_metadata(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def _decode_set_description(self) -> str:
        pass

    @abc.abstractmethod
    def _decode_frame_id_set(self) -> Set[FrameId]:
        pass

    @abc.abstractmethod
    def _decode_sensor_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_camera_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_lidar_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_radar_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        pass

    def get_sensor_names(self) -> List[str]:
        _unique_cache_key = self.get_unique_id(extra="sensor_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_sensor_names()),
        )

    def get_camera_names(self) -> List[str]:
        _unique_cache_key = self.get_unique_id(extra="camera_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_camera_names()),
        )

    def get_lidar_names(self) -> List[str]:
        _unique_cache_key = self.get_unique_id(extra="lidar_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_lidar_names()),
        )

    def get_radar_names(self) -> List[str]:
        _unique_cache_key = self.get_unique_id(extra="radar_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_radar_names()),
        )

    def get_frame_ids(self) -> Set[FrameId]:
        _unique_cache_key = self.get_unique_id(extra="frame_ids")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_set(),
        )

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        _unique_cache_key = self.get_unique_id(extra="class_maps")
        class_maps = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_class_maps(),
        )
        return class_maps

    def _decode_camera_sensor(
        self,
        sensor_decoder: CameraSensorDecoder[TDateTime],
    ) -> CameraSensor[TDateTime]:
        sensor = CameraSensor[TDateTime](decoder=sensor_decoder)
        if self.settings.model_decorator is not None:
            sensor = self.settings.model_decorator(sensor)
        return sensor

    def _decode_lidar_sensor(
        self,
        sensor_decoder: LidarSensorDecoder[TDateTime],
    ) -> LidarSensor[TDateTime]:
        sensor = LidarSensor[TDateTime](decoder=sensor_decoder)
        if self.settings.model_decorator is not None:
            sensor = self.settings.model_decorator(sensor)
        return sensor

    def _decode_radar_sensor(
        self,
        sensor_decoder: RadarSensorDecoder[TDateTime],
    ) -> RadarSensor[TDateTime]:
        sensor = RadarSensor[TDateTime](decoder=sensor_decoder)
        if self.settings.model_decorator is not None:
            sensor = self.settings.model_decorator(sensor)
        return sensor

    def get_camera_sensor(self, camera_name: SensorName) -> CameraSensor[TDateTime]:
        sensor_decoder = self._create_camera_sensor_decoder(
            sensor_name=camera_name,
        )
        return self._decode_camera_sensor(sensor_decoder=sensor_decoder)

    def get_lidar_sensor(self, lidar_name: SensorName) -> LidarSensor[TDateTime]:
        sensor_decoder = self._create_lidar_sensor_decoder(
            sensor_name=lidar_name,
        )
        return self._decode_lidar_sensor(sensor_decoder=sensor_decoder)

    def get_radar_sensor(self, radar_name: SensorName) -> RadarSensor[TDateTime]:
        sensor_decoder = self._create_radar_sensor_decoder(
            sensor_name=radar_name,
        )
        return self._decode_radar_sensor(sensor_decoder=sensor_decoder)

    @abc.abstractmethod
    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_radar_sensor_decoder(self, sensor_name: SensorName) -> RadarSensorDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[TDateTime]:
        pass

    def get_frame(
        self,
        frame_id: FrameId,
    ) -> Frame[TDateTime]:
        frame_decoder = self._create_frame_decoder(frame_id=frame_id)
        frame = Frame(decoder=frame_decoder)
        if self.settings.model_decorator is not None:
            frame = self.settings.model_decorator(frame)
        return frame

    @abc.abstractmethod
    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, TDateTime]:
        pass

    def get_frame_id_to_date_time_map(self) -> Dict[FrameId, TDateTime]:
        _unique_cache_key = self.get_unique_id(extra="frame_id_to_date_time_map")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_to_date_time_map(),
        )

    def clear_from_cache(self):
        prefix = self.get_unique_id()
        self.lazy_load_cache.clear_prefix(prefix=prefix)
