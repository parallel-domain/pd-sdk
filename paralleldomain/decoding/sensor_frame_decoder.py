import abc
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np

from paralleldomain.decoding.common import create_cache_key
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import ImageData, PointCloudData, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SensorFrameSetName, SensorName
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache

T = TypeVar("T")


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
