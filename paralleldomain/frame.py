from .sensor import Sensor


class Frame:
    def __init__(self):
        self._sensors: Dict[str, Sensor] = {}

    def add_sensor(self, sensor: Sensor):
        self._sensors[sensor.name] = sensor
