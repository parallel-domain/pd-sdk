from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, cast

from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensor, LidarSensor, Sensor, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class SceneDecoderProtocol(Protocol):
    def get_unique_scene_id(self, scene_name: SceneName) -> str:
        pass

    def decode_scene_description(self, scene_name: SceneName) -> str:
        pass

    def decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        pass

    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    def decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    def decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    def decode_sensor_frame(self, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        pass

    def decode_ego_frame(self, scene_name: SceneName, frame_id: FrameId) -> EgoFrame:
        pass

    def decode_available_sensor_names(self, scene_name: SceneName, frame_id: FrameId) -> List[SensorName]:
        pass

    def decode_sensor(
        self,
        scene_name: SceneName,
        sensor_name: SensorName,
        sensor_frame_factory: Callable[[FrameId, SensorName], SensorFrame],
    ) -> Sensor:
        pass


class Scene:
    def __init__(self, name: SceneName, description: str, decoder: SceneDecoderProtocol):
        self._name = name
        self._unique_cache_key = decoder.get_unique_scene_id(scene_name=name)
        self._description = description
        self._decoder = decoder

    def _load_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        return LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-{frame_id}-{sensor_name}-SensorFrame",
            loader=lambda: self._decoder.decode_sensor_frame(
                scene_name=self.name, frame_id=frame_id, sensor_name=sensor_name
            ),
        )

    def _load_ego_frame(self, frame_id: FrameId) -> EgoFrame:
        return self._decoder.decode_ego_frame(scene_name=self.name, frame_id=frame_id)

    def _load_frame_sensors_name(self, frame_id: FrameId) -> List[SensorName]:
        return LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-{frame_id}-available_sensors",
            loader=lambda: self._decoder.decode_available_sensor_names(scene_name=self.name, frame_id=frame_id),
        )

    def _load_frame_camera_sensors(self, frame_id: FrameId) -> List[SensorName]:
        all_frame_sensors = self._load_frame_sensors_name(frame_id=frame_id)
        camera_sensors = self.camera_names
        return list(set(all_frame_sensors) & set(camera_sensors))

    def _load_frame_lidar_sensors(self, frame_id: FrameId) -> List[SensorName]:
        all_frame_sensors = self._load_frame_sensors_name(frame_id=frame_id)
        lidar_sensors = self.lidar_names
        return list(set(all_frame_sensors) & set(lidar_sensors))

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def frames(self) -> List[Frame]:
        return [self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids]

    @property
    def frame_ids(self) -> List[str]:
        return sorted(self.frame_id_to_date_time_map, key=self.frame_id_to_date_time_map.get)

    @property
    def frame_id_to_date_time_map(self) -> Dict[FrameId, datetime]:
        return LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-frame_id_to_date_time_map",
            loader=lambda: self._decoder.decode_frame_id_to_date_time_map(scene_name=self.name),
        )

    def get_frame(self, frame_id: FrameId) -> Frame:
        return Frame(
            frame_id=frame_id,
            date_time=self.frame_id_to_date_time_map[frame_id],
            sensor_frame_loader=self._load_sensor_frame,
            available_sensors_loader=self._load_frame_sensors_name,
            available_cameras_loader=self._load_frame_camera_sensors,
            available_lidars_loader=self._load_frame_lidar_sensors,
            ego_frame_loader=self._load_ego_frame,
        )

    @property
    def sensors(self) -> List[Sensor]:
        return [self.get_sensor(sensor_name=sensor_name) for sensor_name in self.sensor_names]

    @property
    def cameras(self) -> List[CameraSensor]:
        return [cast(self.get_sensor(sensor_name=sensor_name), CameraSensor) for sensor_name in self.camera_names]

    @property
    def lidars(self) -> List[LidarSensor]:
        return [cast(self.get_sensor(sensor_name=sensor_name), LidarSensor) for sensor_name in self.lidar_names]

    @property
    def sensor_names(self) -> List[str]:
        return LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-sensor_names",
            loader=lambda: self._decoder.decode_sensor_names(scene_name=self.name),
        )

    @property
    def camera_names(self) -> List[str]:
        return LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-camera_names",
            loader=lambda: self._decoder.decode_camera_names(scene_name=self.name),
        )

    @property
    def lidar_names(self) -> List[str]:
        return LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-lidar_names",
            loader=lambda: self._decoder.decode_lidar_names(scene_name=self.name),
        )

    def get_sensor(self, sensor_name: SensorName) -> Sensor:
        return self._decoder.decode_sensor(
            scene_name=self.name, sensor_name=sensor_name, sensor_frame_factory=self._load_sensor_frame
        )

    @staticmethod
    def from_decoder(scene_name: SceneName, decoder: SceneDecoderProtocol) -> "Scene":
        description = decoder.decode_scene_description(scene_name=scene_name)
        return Scene(name=scene_name, description=description, decoder=decoder)
