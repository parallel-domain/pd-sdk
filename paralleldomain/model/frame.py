from datetime import datetime
from typing import Dict, List, Callable, Tuple, Optional

from paralleldomain.model.ego import EgoFrame
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName


class Frame:
    def __init__(
        self,
        frame_id: FrameId,
        date_time: datetime,
        sensor_frame_loader: Callable[[FrameId, SensorName], SensorFrame],
        available_sensors_loader: Callable[[FrameId], List[SensorName]],
        ego_frame_loader: Callable[[FrameId], EgoFrame],
        available_cameras_loader: Callable[[FrameId], List[SensorName]],
        available_lidars_loader: Callable[[FrameId], List[SensorName]],
    ):
        self._ego_frame_loader = ego_frame_loader
        self._sensor_frame_loader = sensor_frame_loader
        self._available_sensors_loader = available_sensors_loader
        self._available_cameras_loader = available_cameras_loader
        self._available_lidars_loader = available_lidars_loader
        self._frame_id = frame_id
        self._date_time = date_time

    @property
    def frame_id(self) -> FrameId:
        return self._frame_id

    @property
    def date_time(self) -> datetime:
        return self._date_time

    @property
    def ego_frame(self) -> EgoFrame:
        return self._ego_frame_loader(self.frame_id)

    def get_sensor(self, sensor_name: SensorName) -> SensorFrame:
        return self._sensor_frame_loader(self.frame_id, sensor_name)

    @property
    def sensor_names(self) -> List[SensorName]:
        return self._available_sensors_loader(self.frame_id)

    @property
    def camera_names(self) -> List[SensorName]:
        return self._available_cameras_loader(self.frame_id)

    @property
    def lidar_names(self) -> List[SensorName]:
        return self._available_lidars_loader(self.frame_id)

    @property
    def sensor_frames(self) -> List[SensorFrame]:
        return [self.get_sensor(sensor_name=name) for name in self.sensor_names]

    @property
    def camera_frames(self) -> List[SensorFrame]:
        return [self.get_sensor(sensor_name=name) for name in self.camera_names]

    @property
    def lidar_frames(self) -> List[SensorFrame]:
        return [self.get_sensor(sensor_name=name) for name in self.lidar_names]
