from __future__ import annotations

import abc
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar, Union

from paralleldomain.decoding.common import DecoderSettings, create_cache_key
from paralleldomain.decoding.scene_access_decoder import SceneAccessDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

if TYPE_CHECKING:
    from paralleldomain.decoding.decoder import SceneDecoder


T = TypeVar("T")

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class FrameDecoder(Generic[TDateTime], SceneAccessDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
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
        self.frame_id = frame_id
        self.dataset_name = dataset_name

    def get_unique_frame_id(self, sensor_name: SensorName = None, extra: Optional[str] = None) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            extra=extra,
        )

    def get_sensor_names(self) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(extra="available_sensors_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_available_sensor_names()),
        )

    def get_camera_names(self) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(extra="available_camera_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_available_camera_names()),
        )

    def get_lidar_names(self) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(extra="available_lidar_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_available_lidar_names()),
        )

    def get_radar_names(self) -> List[SensorName]:
        _unique_cache_key = self.get_unique_frame_id(extra="available_radar_names")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: sorted(self._decode_available_radar_names()),
        )

    def get_ego_frame(self) -> EgoFrame:
        def _cached_pose_load() -> EgoPose:
            _unique_cache_key = self.get_unique_frame_id(extra="ego_pose")
            return self.lazy_load_cache.get_item(
                key=_unique_cache_key,
                loader=lambda: self._decode_ego_pose(),
            )

        return EgoFrame(pose_loader=_cached_pose_load)

    @abc.abstractmethod
    def _decode_ego_pose(self) -> EgoPose:
        pass

    @abc.abstractmethod
    def _decode_available_sensor_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_available_camera_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_available_lidar_names(self) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def _decode_available_radar_names(self) -> List[SensorName]:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        _unique_cache_key = self.get_unique_frame_id(extra="metadata")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_metadata(),
        )

    @abc.abstractmethod
    def _decode_metadata(self) -> Dict[str, Any]:
        pass

    def get_date_time(self) -> TDateTime:
        # if needed add caching here
        return self._decode_datetime()

    @abc.abstractmethod
    def _decode_datetime(self) -> TDateTime:
        pass

    @abc.abstractmethod
    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[TDateTime]:
        pass

    def _decode_camera_sensor_frame(self, decoder: CameraSensorFrameDecoder[TDateTime]) -> CameraSensorFrame[TDateTime]:
        sensor_frame = CameraSensorFrame[TDateTime](decoder=decoder)
        if self.settings.model_decorator is not None:
            sensor_frame = self.settings.model_decorator(sensor_frame)
        return sensor_frame

    @abc.abstractmethod
    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[TDateTime]:
        pass

    @abc.abstractmethod
    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[TDateTime]:
        pass

    def _decode_lidar_sensor_frame(self, decoder: LidarSensorFrameDecoder[TDateTime]) -> LidarSensorFrame[TDateTime]:
        sensor_frame = LidarSensorFrame[TDateTime](decoder=decoder)
        if self.settings.model_decorator is not None:
            sensor_frame = self.settings.model_decorator(sensor_frame)
        return sensor_frame

    def _decode_radar_sensor_frame(self, decoder: RadarSensorFrameDecoder[TDateTime]) -> RadarSensorFrame[TDateTime]:
        sensor_frame = RadarSensorFrame[TDateTime](decoder=decoder)
        if self.settings.model_decorator is not None:
            sensor_frame = self.settings.model_decorator(sensor_frame)
        return sensor_frame

    def get_camera_sensor_frame(self, sensor_name: SensorName) -> CameraSensorFrame[TDateTime]:
        _unique_cache_key = self.get_unique_frame_id(sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_camera_sensor_frame(
                decoder=self._create_camera_sensor_frame_decoder(sensor_name=sensor_name),
            ),
        )

    def get_lidar_sensor_frame(self, sensor_name: SensorName) -> LidarSensorFrame[TDateTime]:
        _unique_cache_key = self.get_unique_frame_id(sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_lidar_sensor_frame(
                decoder=self._create_lidar_sensor_frame_decoder(sensor_name=sensor_name),
            ),
        )

    def get_radar_sensor_frame(self, sensor_name: SensorName) -> RadarSensorFrame[TDateTime]:
        _unique_cache_key = self.get_unique_frame_id(sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_radar_sensor_frame(
                decoder=self._create_radar_sensor_frame_decoder(sensor_name=sensor_name),
            ),
        )
