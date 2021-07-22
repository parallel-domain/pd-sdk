import contextlib
from datetime import datetime
from typing import Any, Callable, ContextManager, Dict, List, cast

from paralleldomain.model.ego import EgoFrame
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

    def decode_scene_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
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
    def __init__(self, name: SceneName, description: str, metadata: Dict[str, Any], decoder: SceneDecoderProtocol):
        self._name = name
        self._unique_cache_key = decoder.get_unique_scene_id(scene_name=name)
        self._description = description
        self._metadata = metadata
        self._decoder = decoder
        self._cache_is_locked = False
        self._removed_sensor_names = set()

    def lock_cache_for_scene_data(self):
        LAZY_LOAD_CACHE.lock_prefix(prefix=self._unique_cache_key)
        self._cache_is_locked = True

    def unlock_cache_for_scene_data(self):
        LAZY_LOAD_CACHE.unlock_prefix(prefix=self._unique_cache_key)
        self._cache_is_locked = False

    @contextlib.contextmanager
    def editable(self) -> ContextManager["Scene"]:
        self.lock_cache_for_scene_data()
        yield self
        self.unlock_cache_for_scene_data()

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
        temp_sensor_names = LAZY_LOAD_CACHE.get_item(
            key=f"{self._unique_cache_key}-{frame_id}-available_sensors",
            loader=lambda: self._decoder.decode_available_sensor_names(scene_name=self.name, frame_id=frame_id),
        )

        return [sn for sn in temp_sensor_names if sn not in self._removed_sensor_names]

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
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

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
        return [cast(CameraSensor, self.get_sensor(sensor_name=sensor_name)) for sensor_name in self.camera_names]

    @property
    def lidars(self) -> List[LidarSensor]:
        return [cast(LidarSensor, self.get_sensor(sensor_name=sensor_name)) for sensor_name in self.lidar_names]

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

    def remove_sensor(self, sensor_name: SensorName):
        if not self._cache_is_locked:
            sx_msg = (
                "In order to make sure changes are not removed in the cache you need to call "
                "lock_cache_for_scene_data in order to keep those changes from being removed!"
            )
            raise Exception(sx_msg)
        self.sensor_names.remove(sensor_name)
        self._removed_sensor_names.add(sensor_name)

    @staticmethod
    def from_decoder(scene_name: SceneName, decoder: SceneDecoderProtocol) -> "Scene":
        description = decoder.decode_scene_description(scene_name=scene_name)
        metadata = decoder.decode_scene_metadata(scene_name=scene_name)
        return Scene(name=scene_name, description=description, metadata=metadata, decoder=decoder)
