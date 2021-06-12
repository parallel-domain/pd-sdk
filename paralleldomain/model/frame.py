from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName


class Frame:
    def __init__(
        self,
        frame_id: FrameId,
        date_time: datetime,
        sensor_frame_loader: Callable[[FrameId, SensorName], SensorFrame],
        available_sensors_loader: Callable[[FrameId], List[SensorName]],
    ):
        self._sensor_frame_loader = sensor_frame_loader
        self._available_sensors_loader = available_sensors_loader
        self._frame_id = frame_id
        self._date_time = date_time

    @property
    def frame_id(self) -> FrameId:
        return self._frame_id

    @property
    def date_time(self) -> datetime:
        return self._date_time

    def get_sensor(self, sensor_name: SensorName) -> SensorFrame:
        return self._sensor_frame_loader(self.frame_id, sensor_name)

    @property
    def available_sensors(self) -> List[str]:
        return self._available_sensors_loader(self.frame_id)
