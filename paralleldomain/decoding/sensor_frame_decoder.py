import abc
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

from paralleldomain.decoding.common import DecoderSettings, LazyLoadPropertyMixin, create_cache_key
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose, SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookup

T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class SensorFrameDecoder(Generic[TDateTime], LazyLoadPropertyMixin):
    def __init__(self, dataset_name: str, scene_name: SceneName, settings: DecoderSettings):
        self.settings = settings
        self.scene_name = scene_name
        self.dataset_name = dataset_name

    def get_unique_sensor_frame_id(
        self, sensor_name: Optional[SensorName], frame_id: Optional[FrameId], extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=frame_id,
            sensor_name=sensor_name,
            extra=extra,
        )

    def get_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="extrinsic"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_extrinsic(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="sensor_pose"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_sensor_pose(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_annotations(self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier[T]) -> T:
        if self.settings.cache_annotations:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra=f"-annotations-{identifier}"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_annotations(
                    sensor_name=sensor_name, frame_id=frame_id, identifier=identifier
                ),
            )
        else:
            return self._decode_annotations(sensor_name=sensor_name, frame_id=frame_id, identifier=identifier)

    def get_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra=f"-file_path-{data_type.__name__}"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_file_path(sensor_name=sensor_name, frame_id=frame_id, data_type=data_type),
        )

    def get_available_annotation_identifiers(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> List[AnnotationIdentifier]:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="-available_annotation_types"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_available_annotation_identifiers(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="-metadata"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_metadata(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        _unique_cache_key = self.get_unique_sensor_frame_id(extra="class_maps", sensor_name=None, frame_id=None)
        class_maps = self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_class_maps(),
        )
        return class_maps

    def get_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> TDateTime:
        # if needed add caching here
        return self._decode_date_time(sensor_name=sensor_name, frame_id=frame_id)

    @abc.abstractmethod
    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        pass

    @abc.abstractmethod
    def _decode_available_annotation_identifiers(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> List[AnnotationIdentifier]:
        pass

    @abc.abstractmethod
    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> TDateTime:
        pass

    @abc.abstractmethod
    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        pass

    @abc.abstractmethod
    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        pass

    @abc.abstractmethod
    def _decode_annotations(self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier[T]) -> T:
        pass

    @abc.abstractmethod
    def _decode_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        pass


class CameraSensorFrameDecoder(SensorFrameDecoder[TDateTime]):
    def get_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="intrinsic"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_intrinsic(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_distortion_lookup(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[DistortionLookup]:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="distortion_lookup"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_distortion_lookup(sensor_name=sensor_name, frame_id=frame_id),
        )

    def get_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        if self.settings.cache_images:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="image-dims"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_image_dimensions(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_image_dimensions(sensor_name=sensor_name, frame_id=frame_id)

    def get_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        if self.settings.cache_images:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="image_rgba"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_image_rgba(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_image_rgba(sensor_name=sensor_name, frame_id=frame_id)

    @abc.abstractmethod
    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        pass

    def _decode_distortion_lookup(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[DistortionLookup]:
        if sensor_name in self.settings.distortion_lookups:
            return self.settings.distortion_lookups[sensor_name]
        return None

    @abc.abstractmethod
    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        pass

    @abc.abstractmethod
    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def _decode_image_path(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[AnyPath]:
        return None

    def get_image_path(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[AnyPath]:
        _unique_cache_key = self.get_unique_sensor_frame_id(
            sensor_name=sensor_name, frame_id=frame_id, extra="image_path"
        )
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_image_path(sensor_name=sensor_name, frame_id=frame_id),
        )


class LidarSensorFrameDecoder(SensorFrameDecoder[TDateTime]):
    def get_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_size"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_xyz"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_xyz(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_xyz(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_rgb"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_rgb(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_rgb(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_intensity"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_intensity(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_intensity(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_elongation(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_elongation"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_elongation(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_elongation(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_timestamp"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_timestamp(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_timestamp(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_ring_index"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_ring_index(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_ring_index(sensor_name=sensor_name, frame_id=frame_id)

    def get_point_cloud_ray_type(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="point_cloud_ray_type"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_point_cloud_ray_type(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_point_cloud_ray_type(sensor_name=sensor_name, frame_id=frame_id)

    @abc.abstractmethod
    def _decode_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_elongation(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_point_cloud_ray_type(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass


class RadarSensorFrameDecoder(SensorFrameDecoder[TDateTime]):
    def get_radar_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_size"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_xyz"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_xyz(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_xyz(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_rgb"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_rgb(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_rgb(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_power(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_power"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_power(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_power(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_rcs(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_rcs"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_rcs(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_rcs(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_range(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_range"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_range(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_range(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_azimuth(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_azimuth"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_azimuth(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_azimuth(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_elevation(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_elevation"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_elevation(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_elevation(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_timestamp"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_timestamp(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_timestamp(sensor_name=sensor_name, frame_id=frame_id)

    def get_radar_point_cloud_doppler(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_point_cloud_doppler"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_point_cloud_doppler(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_point_cloud_doppler(sensor_name=sensor_name, frame_id=frame_id)

    def get_range_doppler_energy_map(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        if self.settings.cache_point_clouds:
            _unique_cache_key = self.get_unique_sensor_frame_id(
                sensor_name=sensor_name, frame_id=frame_id, extra="radar_range_doppler_energy_map"
            )
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_radar_range_doppler_energy_map(sensor_name=sensor_name, frame_id=frame_id),
            )
        else:
            return self._decode_radar_range_doppler_energy_map(sensor_name=sensor_name, frame_id=frame_id)

    @abc.abstractmethod
    def _decode_radar_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_power(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_rcs(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_range(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_azimuth(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_elevation(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_point_cloud_doppler(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def _decode_radar_range_doppler_energy_map(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Optional[np.ndarray]:
        pass
