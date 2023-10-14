from __future__ import annotations

import abc
from datetime import datetime
from typing import TYPE_CHECKING, Generic, List, Optional, Set, TypeVar, Union

from paralleldomain.decoding.common import DecoderSettings, LazyLoadPropertyMixin, create_cache_key
from paralleldomain.decoding.scene_access_decoder import SceneAccessDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

if TYPE_CHECKING:
    from paralleldomain.decoding.decoder import SceneDecoder

T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class SensorDecoder(Generic[TDateTime], SceneAccessDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
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
        self.dataset_name = dataset_name
        self.sensor_name = sensor_name

    def get_unique_sensor_id(self, frame_id: Optional[FrameId] = None, extra: Optional[str] = None) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    def get_frame_ids(self) -> Set[FrameId]:
        _unique_cache_key = self.get_unique_sensor_id(extra="frame_ids")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_set(),
        )

    @abc.abstractmethod
    def _decode_frame_id_set(self) -> Set[FrameId]:
        pass


class CameraSensorDecoder(SensorDecoder[TDateTime]):
    def _decode_camera_sensor_frame(self, decoder: CameraSensorFrameDecoder[TDateTime]) -> CameraSensorFrame[TDateTime]:
        sensor_frame = CameraSensorFrame[TDateTime](decoder=decoder)
        if self.settings.model_decorator is not None:
            sensor_frame = self.settings.model_decorator(sensor_frame)
        return sensor_frame

    @abc.abstractmethod
    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[TDateTime]:
        pass

    def get_sensor_frame(self, frame_id: FrameId) -> CameraSensorFrame[TDateTime]:
        unique_sensor_id = self.get_unique_sensor_id(frame_id=frame_id, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_camera_sensor_frame(
                decoder=self._create_camera_sensor_frame_decoder(frame_id=frame_id)
            ),
        )


class LidarSensorDecoder(SensorDecoder[TDateTime]):
    def _decode_lidar_sensor_frame(self, decoder: LidarSensorFrameDecoder[TDateTime]) -> LidarSensorFrame[TDateTime]:
        sensor_frame = LidarSensorFrame[TDateTime](decoder=decoder)
        if self.settings.model_decorator is not None:
            sensor_frame = self.settings.model_decorator(sensor_frame)
        return sensor_frame

    @abc.abstractmethod
    def _create_lidar_sensor_frame_decoder(self, frame_id: FrameId) -> LidarSensorFrameDecoder[TDateTime]:
        pass

    def get_sensor_frame(self, frame_id: FrameId) -> LidarSensorFrame[TDateTime]:
        unique_sensor_id = self.get_unique_sensor_id(frame_id=frame_id, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_lidar_sensor_frame(
                decoder=self._create_lidar_sensor_frame_decoder(frame_id=frame_id)
            ),
        )


class RadarSensorDecoder(SensorDecoder[TDateTime]):
    def _decode_radar_sensor_frame(self, decoder: RadarSensorFrameDecoder[TDateTime]) -> RadarSensorFrame[TDateTime]:
        sensor_frame = RadarSensorFrame[TDateTime](decoder=decoder)
        if self.settings.model_decorator is not None:
            sensor_frame = self.settings.model_decorator(sensor_frame)
        return sensor_frame

    @abc.abstractmethod
    def _create_radar_sensor_frame_decoder(self, frame_id: FrameId) -> RadarSensorFrameDecoder[TDateTime]:
        pass

    def get_sensor_frame(self, frame_id: FrameId) -> RadarSensorFrame[TDateTime]:
        unique_sensor_id = self.get_unique_sensor_id(extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_radar_sensor_frame(
                decoder=self._create_radar_sensor_frame_decoder(frame_id=frame_id)
            ),
        )
