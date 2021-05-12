from typing import Dict

from paralleldomain.sensor import SensorFrame


class Frame:
    def __init__(self):
        self._sensor_frames: Dict[str, SensorFrame] = {}

    @property
    def sensors(self) -> Dict[str, SensorFrame]:
        return self._sensor_frames

    # @property TODO
    # def camera_Sensors(self):
    #     return [sen for sen in self.sensors if isinstance(sen, CameraSensor)]

    def add_sensor(self, sensor_frame: SensorFrame):
        self._sensor_frames[sensor_frame.sensor_name] = sensor_frame
