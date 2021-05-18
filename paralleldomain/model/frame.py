from typing import Dict, List, Callable, Tuple, Optional
import numpy as np

from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName


class Frame:
    def __init__(self, frame_id: FrameId, sensor_frame_loader: Callable[[FrameId, SensorName], SensorFrame],
                 available_sensors_loader: Callable[[FrameId], List[SensorName]]):
        self._sensor_frame_loader = sensor_frame_loader
        self._available_sensors_loader = available_sensors_loader
        self.frame_id = frame_id
        self._sensor_frames: Dict[str, SensorFrame] = {}
        self._available_sensors: Optional[List[SensorName]] = None

    def get_sensor(self, sensor_name: SensorName) -> SensorFrame:
        return self._sensor_frame_loader(self.frame_id, sensor_name)

    @property
    def available_sensors(self) -> List[str]:
        if self._available_sensors is None:
            self._available_sensors = self._available_sensors_loader(self.frame_id)
        return self._available_sensors
