import abc
from datetime import datetime
from typing import Generic, List, Optional, TypeVar, Union

from paralleldomain.decoding.common import LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache

T = TypeVar("T")

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class FrameDecoder(Generic[TDateTime], LazyLoadPropertyMixin):
    def __init__(self, dataset_name: str, scene_name: SceneName):
        self.scene_name = scene_name

        self.dataset_name = dataset_name

    def get_unique_frame_id(
        self, frame_id: Optional[FrameId], sensor_name: SensorName = None, extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
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

    def get_date_time(self, frame_id: FrameId) -> TDateTime:
        # if needed add caching here
        return self._decode_datetime(frame_id=frame_id)

    @abc.abstractmethod
    def _decode_datetime(self, frame_id: FrameId) -> TDateTime:
        pass

    @abc.abstractmethod
    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[TDateTime]:
        pass

    def get_camera_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> CameraSensorFrame[TDateTime]:
        _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_camera_sensor_frame(
                decoder=self._create_camera_sensor_frame_decoder(),
                frame_id=frame_id,
                sensor_name=sensor_name,
            ),
        )

    def get_lidar_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> LidarSensorFrame[TDateTime]:
        _unique_cache_key = self.get_unique_frame_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_lidar_sensor_frame(
                decoder=self._create_lidar_sensor_frame_decoder(),
                frame_id=frame_id,
                sensor_name=sensor_name,
            ),
        )
