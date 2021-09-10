import abc
from datetime import datetime
from typing import Generic, Optional, Set, TypeVar, Union

from paralleldomain.decoding.common import LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class SensorDecoder(Generic[TDateTime], LazyLoadPropertyMixin):
    def __init__(self, dataset_name: str, scene_name: SceneName):
        self.scene_name = scene_name
        self.dataset_name = dataset_name

    def get_unique_sensor_id(
        self, sensor_name: SensorName, frame_id: Optional[FrameId] = None, extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
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


class CameraSensorDecoder(SensorDecoder[TDateTime]):
    @abc.abstractmethod
    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[TDateTime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[TDateTime]:
        pass

    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> CameraSensorFrame[TDateTime]:
        unique_sensor_id = self.get_unique_sensor_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_camera_sensor_frame(
                decoder=self._create_camera_sensor_frame_decoder(),
                frame_id=frame_id,
                camera_name=sensor_name,
            ),
        )


class LidarSensorDecoder(SensorDecoder[TDateTime]):
    @abc.abstractmethod
    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[TDateTime], frame_id: FrameId, lidar_name: SensorName
    ) -> LidarSensorFrame[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[TDateTime]:
        pass

    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> LidarSensorFrame[TDateTime]:
        unique_sensor_id = self.get_unique_sensor_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_lidar_sensor_frame(
                decoder=self._create_lidar_sensor_frame_decoder(),
                frame_id=frame_id,
                lidar_name=sensor_name,
            ),
        )
