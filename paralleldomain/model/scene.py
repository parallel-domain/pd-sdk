from typing import Dict, List, Optional, Tuple

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.type_aliases import FrameId, SensorName, SceneName
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import Sensor, SensorFrame, PointCloudData


class SceneDecoderProtocol(Protocol):
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
        self._description = description
        self._decoder = decoder
        self._frames: Dict[str, Frame] = dict()
        self._sensor_frames: Dict[Tuple[FrameId, SensorName], SensorFrame] = dict()
        self._sensor_names: Optional[List[str]] = None
        self._frame_ids: Optional[List[str]] = None
        self._sensors: Dict[str, Sensor] = dict()
        self._prepared_frames = False

    def _setup_frame(self, frame_id: FrameId):
        if frame_id not in self._frames:
            self._frames[frame_id] = Frame(frame_id=frame_id,
                                           sensor_frame_loader=self._load_sensor_frame,
                                           available_sensors_loader=self._load_available_sensors)

    def _setup_sensor(self, sensor_name: SensorName):
        if sensor_name not in self._sensors:
            self._sensors[sensor_name] = Sensor(sensor_name=sensor_name,
                                                sensor_frame_factory=self._load_sensor_frame)

    def _load_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        key = (frame_id, sensor_name)
        if key not in self._sensor_frames:
            self._sensor_frames[key] = self._decoder.decode_sensor_frame(scene_name=self.name,
                                                                         frame_id=frame_id,
                                                                         sensor_name=sensor_name)
        return self._sensor_frames[key]

    def _load_available_sensors(self, frame_id: FrameId) -> List[SensorName]:
        return self._decoder.decode_available_sensor_names(scene_name=self.name, frame_id=frame_id)

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
        if self._frame_ids is None:
            self._frame_ids = self._decoder.decode_frame_ids(scene_name=self.name)
        return self._frame_ids

    def get_frame(self, frame_id: FrameId) -> Frame:
        if frame_id not in self.frame_ids:
            raise ValueError(f"Unknown FrameId {frame_id}")
        self._setup_frame(frame_id=frame_id)
        return self._frames[frame_id]

    @property
    def sensors(self) -> List[Sensor]:
        return [self.get_sensor(sensor_name=sensor_name) for sensor_name in self.sensor_names]

    @property
    def sensor_names(self) -> List[str]:
        if self._sensor_names is None:
            self._sensor_names = self._decoder.decode_sensor_names(scene_name=self.name)
        return self._sensor_names

    def get_sensor(self, sensor_name: SensorName) -> Sensor:
        self._setup_sensor(sensor_name=sensor_name)
        return self._sensors[sensor_name]

    @staticmethod
    def from_decoder(scene_name: SceneName, decoder: SceneDecoderProtocol) -> "Scene":
        description = decoder.decode_scene_description(scene_name=scene_name)
        return Scene(name=scene_name,
                     description=description,
                     decoder=decoder)