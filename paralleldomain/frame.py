from .sensor import SensorFrame


class Frame:
    def __init__(self):
        self._sensor_frames: Dict[str, SensorFrame] = {}

    @property
    def sensors(self):
        return self._sensor_frames

    def add_sensor(self, sensor_frame: SensorFrame):
        self._sensor_frames[sensor_frame.sensor.name] = sensor_frame
