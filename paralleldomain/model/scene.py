from typing import Dict, List, Optional, Tuple

from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.type_aliases import FrameId, SensorName, SceneName
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import Sensor, SensorFrame


class SceneDecoderProtocol(Protocol):
    def get_unique_scene_id(self, scene_name: SceneName) -> str:
        pass

    def decode_scene_description(self, scene_name: SceneName) -> str:
        pass

    def decode_frame_ids(self, scene_name: SceneName) -> List[FrameId]:
        pass

    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    def decode_sensor_frame(self, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        pass

    def decode_available_sensor_names(self, scene_name: SceneName, frame_id: FrameId) -> List[SensorName]:
        pass


class Scene:
    def __init__(self, name: SceneName, description: str, decoder: SceneDecoderProtocol):
        self._name = name
        self._unique_cache_key = decoder.get_unique_scene_id(scene_name=name)
        self._description = description
        self._decoder = decoder

    def _load_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        return LAZY_LOAD_CACHE.get_item(key=f"{self._unique_cache_key}-{frame_id}-{sensor_name}-SensorFrame",
                                        loader=lambda: self._decoder.decode_sensor_frame(
                                            scene_name=self.name,
                                            frame_id=frame_id,
                                            sensor_name=sensor_name))

    def _load_available_sensors(self, frame_id: FrameId) -> List[SensorName]:
        return LAZY_LOAD_CACHE.get_item(key=f"{self._unique_cache_key}-{frame_id}-available_sensors",
                                        loader=lambda: self._decoder.decode_available_sensor_names(
                                            scene_name=self.name, frame_id=frame_id))

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
        return LAZY_LOAD_CACHE.get_item(key=f"{self._unique_cache_key}-frame_ids",
                                        loader=lambda: self._decoder.decode_frame_ids(scene_name=self.name))

    def get_frame(self, frame_id: FrameId) -> Frame:
        return Frame(frame_id=frame_id,
                     sensor_frame_loader=self._load_sensor_frame,
                     available_sensors_loader=self._load_available_sensors)

    @property
    def sensors(self) -> List[Sensor]:
        return [self.get_sensor(sensor_name=sensor_name) for sensor_name in self.sensor_names]

    @property
    def sensor_names(self) -> List[str]:
        return LAZY_LOAD_CACHE.get_item(key=f"{self._unique_cache_key}-sensor_names",
                                        loader=lambda: self._decoder.decode_sensor_names(scene_name=self.name))

    def get_sensor(self, sensor_name: SensorName) -> Sensor:
        return Sensor(sensor_name=sensor_name,
                      sensor_frame_factory=self._load_sensor_frame)

    @staticmethod
    def from_decoder(scene_name: SceneName, decoder: SceneDecoderProtocol) -> "Scene":
        description = decoder.decode_scene_description(scene_name=scene_name)
        return Scene(name=scene_name,
                     description=description,
                     decoder=decoder)
