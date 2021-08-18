import abc
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar

import numpy as np

from paralleldomain import Scene
from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import Dataset, DatasetMeta, SceneDataset
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import (
    ImageData,
    PointCloudData,
    Sensor,
    SensorExtrinsic,
    SensorFrame,
    SensorIntrinsic,
    SensorPose,
)
from paralleldomain.model.sensor_frame_set import SensorFrameSet
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorFrameSetName, SensorName
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE, LazyLoadCache, cache_max_ram_usage_factor

T = TypeVar("T")

TDatasetType = TypeVar("TDatasetType", bound=Dataset)
TSceneDatasetType = TypeVar("TSceneDatasetType", bound=SceneDataset)
TSensorFrameType = TypeVar("TSensorFrameType", bound=SensorFrame)

# TSensorFrame = TypeVar("TSensorFrame", bound=SensorFrame)


def create_cache_key(
    dataset_name: str,
    set_name: Optional[SensorFrameSetName] = None,
    frame_id: Optional[FrameId] = None,
    sensor_name: Optional[SensorName] = None,
    extra: Optional[str] = None,
) -> str:
    cache_key = f"{dataset_name}"
    if set_name is not None:
        cache_key += f"-{set_name}"
    if frame_id is not None:
        cache_key += f"-{frame_id}"
    if sensor_name is not None:
        cache_key += f"-{sensor_name}"
    if extra is not None:
        cache_key += f"-{extra}"
    return cache_key


class SensorFrameDecoder:
    def __init__(self, dataset_name: str, set_name: SensorFrameSetName, lazy_load_cache: LazyLoadCache):
        self.set_name = set_name
        self._lazy_load_cache = lazy_load_cache
        self.dataset_name = dataset_name

    def get_unique_sensor_frame_id(
        self, sensor_name: SensorName, frame_id: FrameId, extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            set_name=self.set_name,
            frame_id=frame_id,
            sensor_name=sensor_name,
            extra=extra,
        )

    def get_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="extrinsic"
        )
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_extrinsic(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="intrinsic"
        )
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_intrinsic(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="sensor_pose"
        )
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_pose(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_point_cloud(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[PointCloudData]:
        if self._has_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id):
            point_format = self._decode_point_cloud_format(sensor_name=sensor_name, frame_id=frame_id)

            def _cached_load() -> np.ndarray:
                _unique_cache_key = self.get_unique_sensor_frame_id(
                    sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_data"
                )
                return self._lazy_load_cache.get_item(
                    key=_unique_cache_key,
                    loader=lambda: self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id),
                )

            return PointCloudData(
                point_format=point_format,
                load_data=_cached_load,
            )
        return None

        # return self._lazy_load_cache.get_item(
        #     key=_unique_cache_key + "-point_cloud",
        #     loader=lambda: self._decode_point_cloud(sensor_name=sensor_name, frame_id=frame_id),
        # )

    def get_image(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[ImageData]:
        if self._has_image_data(sensor_name=sensor_name, frame_id=frame_id):

            def _cached_data_load() -> np.ndarray:
                _unique_cache_key = self.get_unique_sensor_frame_id(
                    sensor_name=sensor_name, frame_id=frame_id, extra="mage-data"
                )
                return self._lazy_load_cache.get_item(
                    key=_unique_cache_key,
                    loader=lambda: self._decode_image_data(sensor_name=sensor_name, frame_id=frame_id),
                )

            def _cached_dims_load() -> Tuple[int, int, int]:
                _unique_cache_key = self.get_unique_sensor_frame_id(
                    sensor_name=sensor_name, frame_id=frame_id, extra="image-dims"
                )
                return self._lazy_load_cache.get_item(
                    key=_unique_cache_key,
                    loader=lambda: self._decode_image_dims(sensor_name=sensor_name, frame_id=frame_id),
                )

            return ImageData(
                load_data_rgba=_cached_data_load,
                load_image_dims=_cached_dims_load,
            )
        return None

        # _unique_cache_key = self.get_unique_sensor_frame_id(sensor_name=sensor_name, frame_id=frame_id)
        # return self._lazy_load_cache.get_item(
        #     key=_unique_cache_key + "-image",
        #     loader=lambda: self._decode_image(sensor_name=sensor_name, frame_id=frame_id),
        # )

    def get_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra=f"-annotations-{identifier}"
        )
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_annotations(
                sensor_name=sensor_name, frame_id=frame_id, identifier=identifier, annotation_type=annotation_type
            ),
        )

    def get_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="-available_annotation_types"
        )
        return self._lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_annotation_types(sensor_name=sensor_name, frame_id=frame_id),
        )

    @abc.abstractmethod
    def _decode_point_cloud_format(self, sensor_name: SensorName, frame_id: FrameId) -> List[str]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _has_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> bool:
        pass

    @abc.abstractmethod
    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        pass

    @abc.abstractmethod
    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        pass

    @abc.abstractmethod
    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        pass

    @abc.abstractmethod
    def _decode_image_dims(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        pass

    @abc.abstractmethod
    def _decode_image_data(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _has_image_data(self, sensor_name: SensorName, frame_id: FrameId) -> bool:
        pass

    @abc.abstractmethod
    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> List[T]:
        pass

    @abc.abstractmethod
    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        pass


class TemporalSensorFrameDecoder(SensorFrameDecoder):
    def get_datetime(self, frame_id: FrameId) -> datetime:
        # if needed add caching here
        return self._decode_datetime(frame_id=frame_id)

    @abc.abstractmethod
    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        pass


class SensorDecoder:
    def __init__(self, dataset_name: str, set_name: SensorFrameSetName, lazy_load_cache: LazyLoadCache):
        self.set_name = set_name
        self.lazy_load_cache = lazy_load_cache
        self.dataset_name = dataset_name

    def get_unique_sensor_id(
        self, sensor_name: SensorName, frame_id: Optional[FrameId] = None, extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            set_name=self.set_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    def get_frame_ids(self, sensor_name: SensorName) -> Set[FrameId]:
        _unique_cache_key = self.get_unique_sensor_id(sensor_name=sensor_name, extra="frame_ids")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_set(sensor_name=sensor_name),
        )

    @abc.abstractmethod
    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        pass

    @abc.abstractmethod
    def _decode_sensor_frame(
        self, decoder: SensorFrameDecoder, frame_id: FrameId, sensor_name: SensorName
    ) -> SensorFrame:
        pass

    @abc.abstractmethod
    def _create_sensor_frame_decoder(self) -> SensorFrameDecoder:
        pass

    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        unique_sensor_id = self.get_unique_sensor_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_sensor_frame(
                decoder=self._create_sensor_frame_decoder(),
                frame_id=frame_id,
                sensor_name=sensor_name,
            ),
        )


class FrameDecoder:
    def __init__(self, dataset_name: str, set_name: SensorFrameSetName, lazy_load_cache: LazyLoadCache):
        self.set_name = set_name
        self.lazy_load_cache = lazy_load_cache
        self.dataset_name = dataset_name

    def get_unique_frame_id(
        self, frame_id: Optional[FrameId], sensor_name: SensorName = None, extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            set_name=self.set_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    def get_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, extra="available_sensors_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_sensor_names(frame_id=frame_id),
        )

    def get_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, extra="available_camera_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_camera_names(frame_id=frame_id),
        )

    def get_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, extra="available_lidar_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_lidar_names(frame_id=frame_id),
        )

    def get_ego_frame(self, frame_id: FrameId) -> EgoFrame:
        def _cached_pose_load() -> EgoPose:
            _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, extra="ego_pose")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_ego_pose(frame_id=frame_id),
            )

        return EgoFrame(pose_loader=_cached_pose_load)

    @abc.abstractmethod
    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        pass

    @abc.abstractmethod
    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_sensor_frame(
        self, decoder: SensorFrameDecoder, frame_id: FrameId, sensor_name: SensorName
    ) -> SensorFrame:
        pass

    @abc.abstractmethod
    def _create_sensor_frame_decoder(self) -> SensorFrameDecoder:
        pass

    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_frame(
                decoder=self._create_sensor_frame_decoder(),
                frame_id=frame_id,
                sensor_name=sensor_name,
            ),
        )

    # def _load_frame_sensors_name(self, set_name: SensorFrameSetName, frame_id: FrameId) -> List[SensorName]:
    #     _unique_cache_key = self.get_unique_set_id(set_name=set_name)
    #     return self._lazy_load_cache.get_item(
    #         key=f"{_unique_cache_key}-{frame_id}-available_sensors",
    #         loader=lambda: self._decode_available_sensor_names(set_name=set_name, frame_id=frame_id),
    #     )

    # def _load_frame_camera_sensors(self, set_name: SensorFrameSetName, frame_id: FrameId) -> List[SensorName]:
    #     all_frame_sensors = self._load_frame_sensors_name(set_name=set_name, frame_id=frame_id)
    #     camera_sensors = self.get_camera_names(set_name=set_name)
    #     return list(set(all_frame_sensors) & set(camera_sensors))
    #
    # def _load_frame_lidar_sensors(self, set_name: SensorFrameSetName, frame_id: FrameId) -> List[SensorName]:
    #     all_frame_sensors = self._load_frame_sensors_name(set_name=set_name, frame_id=frame_id)
    #     lidar_sensors = self.get_lidar_names(set_name=set_name)
    #     return list(set(all_frame_sensors) & set(lidar_sensors))


class TemporalFrameDecoder(FrameDecoder):
    def get_datetime(self, frame_id: FrameId) -> datetime:
        # if needed add caching here
        return self._decode_datetime(frame_id=frame_id)

    @abc.abstractmethod
    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        pass


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

    # @abc.abstractmethod
    # def _decode_sensor_frame(
    #     self, set_name: SensorFrameSetName, frame_id: FrameId, sensor_name: SensorName
    # ) -> SensorFrame:
    #     pass

    # @abc.abstractmethod
    # def decode_ego_frame(self, set_name: SensorFrameSetName, frame_id: FrameId) -> EgoFrame:
    #     pass

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
        # return TemporalFrame(
        #     frame_id=frame_id,
        #     date_time=self.frame_id_to_date_time_map[frame_id],
        #     sensor_frame_loader=self._load_sensor_frame,
        #     available_sensors_loader=self._load_frame_sensors_name,
        #     available_cameras_loader=self._load_frame_camera_sensors,
        #     available_lidars_loader=self._load_frame_lidar_sensors,
        #     ego_frame_loader=lambda: self._decode_ego_frame(set_name=set_name, frame_id=frame_id),
        # )


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
