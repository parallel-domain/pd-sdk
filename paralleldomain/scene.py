from .dto import SceneDTO, CalibrationDTO
from .frame import Frame
from .sensor import Sensor, SensorFrame, SensorExtrinsic, SensorIntrinsic
from typing import Dict
import ujson as json


class Scene:
    def __init__(self, scene_dto: SceneDTO, dataset):
        self._dto = scene_dto
        self._dataset = dataset
        self._frames = []
        self._prepare_frames()

    def _data_by_key(self):
        return {d.key: d for d in self._dto.data}

    def _prepare_frames(self):  # quick implementation, tbd better
        sensors = {}
        data = self._data_by_key()
        for sample in self._dto.samples:
            frame = Frame()
            with open(
                f"{self._path}/{self.name}/calibration/{sample.calibration_key}.json",
                "r",
            ) as f:
                calibration = CalibrationDTO.from_dict(json.load(f))

            extrinsics_by_sensor = dict(
                zip(
                    calibration.names,
                    map(
                        SensorExtrinsic.from_CalibrationExtrinsicDTO,
                        calibration.extrinsics,
                    ),
                )
            )

            intrinsics_by_sensor = dict(
                zip(
                    calibration.names,
                    map(
                        SensorIntrinsic.from_CalibrationIntrinsicDTO,
                        calibration.intrinsics,
                    ),
                )
            )

            for key in sample.datum_keys:
                data_row = data[key]
                sensor_name = data_row.id.name
                sensor = (
                    sensors[sensor_name]
                    if sensor_name in sensors.keys()
                    else Sensor(sensor_name)
                )
                sensor_frame = SensorFrame.from_SceneDataDatumDTO(
                    data_row.datum,
                )
                sensor_frame.extrinsic = extrinsics_by_sensor[sensor_name]
                sensor_frame.intrinsic = intrinsics_by_sensor[sensor_name]
                sensor.add_sensor_frame(sensor_frame)

                sensors[sensor_name] = sensor
                frame.add_sensor(sensor)

            self._frames.append(frame)

    @property
    def _path(self):
        return self._dataset._path

    @property
    def name(self):
        return self._dto.name

    @property
    def description(self):
        return self._dto.description

    @staticmethod
    def from_dict(scene_data: Dict, dataset):
        scene = Scene(SceneDTO.from_dict(scene_data), dataset)
        return scene
