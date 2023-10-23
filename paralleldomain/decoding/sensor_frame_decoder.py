from __future__ import annotations

import abc
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

from paralleldomain.decoding.common import DecoderSettings, LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.scene_access_decoder import SceneAccessDecoder
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookup

if TYPE_CHECKING:
    from paralleldomain.decoding.decoder import SceneDecoder
T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class SensorFrameDecoder(Generic[TDateTime], SceneAccessDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder: SceneDecoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            settings=settings,
            is_unordered_scene=is_unordered_scene,
            scene_name=scene_name,
            scene_decoder=scene_decoder,
        )
        self.settings = settings
        self.scene_name = scene_name
        self.sensor_name = sensor_name
        self.frame_id = frame_id
        self.dataset_name = dataset_name

    def get_unique_sensor_frame_id(self, extra: Optional[str] = None) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=self.sensor_name,
            extra=extra,
        )

    def get_extrinsic(self) -> SensorExtrinsic:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="extrinsic")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_extrinsic(),
        )

    def get_sensor_pose(self) -> SensorPose:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="sensor_pose")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_pose(),
        )

    def get_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        if self.settings.cache_annotations:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra=f"-annotations-{identifier}")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_annotations(identifier=identifier),
            )
        else:
            return self._decode_annotations(identifier=identifier)

    def get_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        # Note: We also support Type[Annotation] for data_type for backwards compatibility
        _unique_cache_key = self.get_unique_sensor_frame_id(extra=f"-file_path-{data_type.__name__}")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_file_path(data_type=data_type),
        )

    def get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="-available_annotation_types")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_annotation_identifiers(),
        )

    def get_metadata(self) -> Dict[str, Any]:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="-metadata")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_metadata(),
        )

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        _unique_cache_key = create_cache_key(
            extra="class_maps",
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=None,
            frame_id=None,
        )
        class_maps = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_class_maps(),
        )
        return class_maps

    def get_date_time(self) -> TDateTime:
        # if needed add caching here
        return self._decode_date_time()

    @abc.abstractmethod
    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        pass

    @abc.abstractmethod
    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        pass

    @abc.abstractmethod
    def _decode_metadata(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def _decode_date_time(self) -> TDateTime:
        pass

    @abc.abstractmethod
    def _decode_extrinsic(self) -> SensorExtrinsic:
        pass

    @abc.abstractmethod
    def _decode_sensor_pose(self) -> SensorPose:
        pass

    @abc.abstractmethod
    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        pass

    @abc.abstractmethod
    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        pass


class CameraSensorFrameDecoder(SensorFrameDecoder[TDateTime]):
    def get_intrinsic(self) -> SensorIntrinsic:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="intrinsic")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_intrinsic(),
        )

    def get_distortion_lookup(self) -> Optional[DistortionLookup]:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="distortion_lookup")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_distortion_lookup(),
        )

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        if self.settings.cache_images:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="image-dims")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_image_dimensions(),
            )
        else:
            return self._decode_image_dimensions()

    def get_image_rgba(self) -> np.ndarray:
        if self.settings.cache_images:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="image_rgba")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_image_rgba(),
            )
        else:
            return self._decode_image_rgba()

    @abc.abstractmethod
    def _decode_intrinsic(self) -> SensorIntrinsic:
        pass

    def _decode_distortion_lookup(self) -> Optional[DistortionLookup]:
        if self.sensor_name in self.settings.distortion_lookups:
            return self.settings.distortion_lookups[self.sensor_name]
        return None

    @abc.abstractmethod
    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        pass

    @abc.abstractmethod
    def _decode_image_rgba(self) -> np.ndarray:
        pass

    def _decode_image_path(self) -> Optional[AnyPath]:
        return None

    def get_image_path(self) -> Optional[AnyPath]:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="image_path")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_image_path(),
        )


class LidarSensorFrameDecoder(SensorFrameDecoder[TDateTime]):
    def get_point_cloud_size(self) -> int:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_size")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_size(),
            )
        else:
            return self._decode_point_cloud_size()

    def get_point_cloud_xyz(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_xyz")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_xyz(),
            )
        else:
            return self._decode_point_cloud_xyz()

    def get_point_cloud_rgb(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_rgb")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_rgb(),
            )
        else:
            return self._decode_point_cloud_rgb()

    def get_point_cloud_intensity(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_intensity")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_intensity(),
            )
        else:
            return self._decode_point_cloud_intensity()

    def get_point_cloud_elongation(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_elongation")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_elongation(),
            )
        else:
            return self._decode_point_cloud_elongation()

    def get_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_timestamp")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_timestamp(),
            )
        else:
            return self._decode_point_cloud_timestamp()

    def get_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_ring_index")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_ring_index(),
            )
        else:
            return self._decode_point_cloud_ring_index()

    def get_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="point_cloud_ray_type")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_ray_type(),
            )
        else:
            return self._decode_point_cloud_ray_type()

    @abc.abstractmethod
    def _decode_point_cloud_size(self) -> int:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_xyz(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_rgb(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_intensity(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_elongation(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        pass


class RadarSensorFrameDecoder(SensorFrameDecoder[TDateTime]):
    def get_radar_point_cloud_size(self) -> int:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_size")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_size(),
            )
        else:
            return self._decode_radar_point_cloud_size()

    def get_radar_point_cloud_xyz(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_xyz")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_xyz(),
            )
        else:
            return self._decode_radar_point_cloud_xyz()

    def get_radar_point_cloud_rgb(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_rgb")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_rgb(),
            )
        else:
            return self._decode_radar_point_cloud_rgb()

    def get_radar_point_cloud_power(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_power")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_power(),
            )
        else:
            return self._decode_radar_point_cloud_power()

    def get_radar_point_cloud_rcs(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_rcs")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_rcs(),
            )
        else:
            return self._decode_radar_point_cloud_rcs()

    def get_radar_point_cloud_range(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_range")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_range(),
            )
        else:
            return self._decode_radar_point_cloud_range()

    def get_radar_point_cloud_azimuth(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_azimuth")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_azimuth(),
            )
        else:
            return self._decode_radar_point_cloud_azimuth()

    def get_radar_point_cloud_elevation(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_elevation")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_elevation(),
            )
        else:
            return self._decode_radar_point_cloud_elevation()

    def get_radar_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_timestamp")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_timestamp(),
            )
        else:
            return self._decode_radar_point_cloud_timestamp()

    def get_radar_point_cloud_doppler(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_point_cloud_doppler")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_doppler(),
            )
        else:
            return self._decode_radar_point_cloud_doppler()

    def get_range_doppler_energy_map(self) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(extra="radar_range_doppler_energy_map")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_range_doppler_energy_map(),
            )
        else:
            return self._decode_radar_range_doppler_energy_map()

    @abc.abstractmethod
    def _decode_radar_point_cloud_size(self) -> int:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_xyz(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_rgb(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_power(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_rcs(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_range(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_azimuth(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_elevation(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_doppler(self) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_range_doppler_energy_map(self) -> Optional[np.ndarray]:
        pass
