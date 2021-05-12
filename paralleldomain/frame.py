from typing import Dict, List
import numpy as np

from paralleldomain.sensor import SensorFrame


class Frame:
    def __init__(self):
        self._sensor_frames: Dict[str, SensorFrame] = {}

    @property
    def sensors(self) -> Dict[str, SensorFrame]:
        return self._sensor_frames

    @property
    def available_sensors(self) -> List[str]:
        return list(self._sensor_frames.keys())

    def add_sensor(self, sensor_frame: SensorFrame):
        self._sensor_frames[sensor_frame.sensor_name] = sensor_frame

    @property
    def point_clouds(self) -> Dict[str, np.ndarray]:
        # todo load from sensor frame lazyload on dict __getitem__
        pass

    @property
    def images(self) -> Dict[str, np.ndarray]:
        # todo load from sensor frame lazyload on dict __getitem_
        pass