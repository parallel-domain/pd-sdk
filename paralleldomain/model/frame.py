from typing import Dict, List, Callable, Tuple, Optional

from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName


class Frame:
    def __init__(self, frame_id: FrameId, sensor_frame_loader: Callable[[FrameId, SensorName], SensorFrame],
                 available_sensors_loader: Callable[[FrameId], List[SensorName]]):
        self._sensor_frame_loader = sensor_frame_loader
        self._available_sensors_loader = available_sensors_loader
        self.frame_id = frame_id

    def get_sensor(self, sensor_name: SensorName) -> SensorFrame:
        return self._sensor_frame_loader(self.frame_id, sensor_name)

    @property
    def available_sensors(self) -> List[str]:
        return self._available_sensors_loader(self.frame_id)
