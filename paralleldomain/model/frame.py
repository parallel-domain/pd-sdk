from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Generator, Generic, List, Optional, Protocol, TypeVar, Union

from paralleldomain.model.ego import EgoFrame
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

if TYPE_CHECKING:
    from paralleldomain.model.scene import Scene
    from paralleldomain.model.unordered_scene import UnorderedScene

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class FrameDecoderProtocol(Protocol[TDateTime]):
    def get_camera_sensor_frame(self, sensor_name: SensorName) -> CameraSensorFrame[TDateTime]:
        pass

    def get_lidar_sensor_frame(self, sensor_name: SensorName) -> LidarSensorFrame[TDateTime]:
        pass

    def get_radar_sensor_frame(self, sensor_name: SensorName) -> RadarSensorFrame[TDateTime]:
        pass

    def get_sensor_names(self) -> List[SensorName]:
        pass

    def get_camera_names(self) -> List[SensorName]:
        pass

    def get_lidar_names(self) -> List[SensorName]:
        pass

    def get_radar_names(self) -> List[SensorName]:
        pass

    def get_ego_frame(self) -> EgoFrame:
        pass

    def get_date_time(self) -> TDateTime:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        pass

    @property
    def dataset_name(self) -> str:
        pass

    @property
    def scene_name(self) -> SceneName:
        pass

    @property
    def frame_id(self) -> FrameId:
        pass

    def get_scene(self) -> Union[Scene, UnorderedScene]:
        pass


class Frame(Generic[TDateTime]):
    def __init__(
        self,
        decoder: FrameDecoderProtocol[TDateTime],
    ):
        self._decoder = decoder

    @property
    def dataset_name(self) -> str:
        return self._decoder.dataset_name

    @property
    def scene_name(self) -> SceneName:
        return self._decoder.scene_name

    @property
    def scene(self) -> Union[Scene, UnorderedScene]:
        return self._decoder.get_scene()

    @property
    def next_frame(self) -> Optional[Frame[TDateTime]]:
        next_frame_id = self.next_frame_id
        if next_frame_id is not None:
            return self.scene.get_frame(frame_id=next_frame_id)
        return None

    @property
    def previous_frame(self) -> Optional[Frame[TDateTime]]:
        previous_frame_id = self.previous_frame_id
        if previous_frame_id is not None:
            return self.scene.get_frame(frame_id=previous_frame_id)
        return None

    @property
    def next_frame_id(self) -> Optional[FrameId]:
        scene = self.scene
        if not scene.is_ordered:
            return None
        frame_ids = scene.frame_ids
        next_frame_id_idx = frame_ids.index(self.frame_id) + 1
        if 0 <= next_frame_id_idx < len(frame_ids):
            return frame_ids[next_frame_id_idx]
        return None

    @property
    def previous_frame_id(self) -> Optional[FrameId]:
        scene = self.scene
        if not scene.is_ordered:
            return None
        frame_ids = scene.frame_ids
        previous_frame_id_idx = frame_ids.index(self.frame_id) - 1
        if 0 <= previous_frame_id_idx < len(frame_ids):
            return frame_ids[previous_frame_id_idx]
        return None

    @property
    def frame_id(self) -> FrameId:
        return self._decoder.frame_id

    @property
    def date_time(self) -> TDateTime:
        return self._decoder.get_date_time()

    @property
    def ego_frame(self) -> EgoFrame:
        return self._decoder.get_ego_frame()

    def get_camera(self, camera_name: SensorName) -> CameraSensorFrame[TDateTime]:
        if camera_name not in self.camera_names:
            raise ValueError(f"Camera {camera_name} could not be found.")
        return self._decoder.get_camera_sensor_frame(sensor_name=camera_name)

    def get_lidar(self, lidar_name: SensorName) -> LidarSensorFrame[TDateTime]:
        if lidar_name not in self.lidar_names:
            raise ValueError(f"LiDAR {lidar_name} could not be found.")
        return self._decoder.get_lidar_sensor_frame(sensor_name=lidar_name)

    def get_radar(self, radar_name: SensorName) -> RadarSensorFrame[TDateTime]:
        if radar_name not in self.radar_names:
            raise ValueError(f"Radar {radar_name} could not be found.")
        return self._decoder.get_radar_sensor_frame(sensor_name=radar_name)

    def get_sensor(self, sensor_name: SensorName) -> SensorFrame[TDateTime]:
        if sensor_name in self.camera_names:
            return self.get_camera(camera_name=sensor_name)
        elif sensor_name in self.lidar_names:
            return self.get_lidar(lidar_name=sensor_name)
        elif sensor_name in self.radar_names:
            return self.get_radar(radar_name=sensor_name)
        else:
            raise ValueError(f"Sensor {sensor_name} could not be found.")

    def _get_sensor_names(self) -> List[SensorName]:
        return self._decoder.get_sensor_names()

    @property
    def sensor_names(self) -> List[SensorName]:
        return self._get_sensor_names()

    def _get_camera_names(self) -> List[SensorName]:
        return self._decoder.get_camera_names()

    @property
    def camera_names(self) -> List[SensorName]:
        return self._get_camera_names()

    def _get_lidar_names(self) -> List[SensorName]:
        return self._decoder.get_lidar_names()

    @property
    def lidar_names(self) -> List[SensorName]:
        return self._get_lidar_names()

    def _get_radar_names(self) -> List[SensorName]:
        return self._decoder.get_radar_names()

    @property
    def radar_names(self) -> List[SensorName]:
        return self._get_radar_names()

    @property
    def sensor_frames(self) -> Generator[SensorFrame[TDateTime], None, None]:
        return (self.get_sensor(sensor_name=name) for name in self.sensor_names)

    @property
    def camera_frames(self) -> Generator[CameraSensorFrame[TDateTime], None, None]:
        return (self._decoder.get_camera_sensor_frame(sensor_name=name) for name in self.camera_names)

    @property
    def lidar_frames(self) -> Generator[LidarSensorFrame[TDateTime], None, None]:
        return (self._decoder.get_lidar_sensor_frame(sensor_name=name) for name in self.lidar_names)

    @property
    def radar_frames(self) -> Generator[RadarSensorFrame[TDateTime], None, None]:
        return (self._decoder.get_radar_sensor_frame(sensor_name=name) for name in self.radar_names)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._decoder.get_metadata()

    def __lt__(self, other: "Frame[TDateTime]"):
        if self.date_time is not None and other.date_time is not None:
            return self.date_time < other.date_time
        return id(self) < id(other)
