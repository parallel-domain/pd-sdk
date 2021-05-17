from typing import Dict, List
import ujson as json

from paralleldomain.dto import SceneDTO, CalibrationDTO
from paralleldomain.frame import Frame
from paralleldomain.sensor import Sensor, SensorFrame, SensorExtrinsic, SensorIntrinsic


class Scene:
    def __init__(self, scene_dto: SceneDTO, dataset_path: str):
        self._dto = scene_dto
        self._dataset_path = dataset_path
        self._scene_path = f"{self._dataset_path}/{self.name}"
        self._frames: List[Frame] = []
        self._sensors: Dict[str, Sensor] = {}
        self._prepare_frames()

    def _data_by_key(self) -> Dict[str, SceneDTO]:
        return {d.key: d for d in self._dto.data}

    def _add_frame(self, frame: Frame):
        self._frames.append(frame)
        self._sensors.update(
            {sf.sensor.name: sf.sensor for _, sf in frame.sensors.items()}
        )

    def _prepare_frames(self):  # quick implementation, tbd better
        sensors: Dict[str, Sensor] = dict()
        data = self._data_by_key()
        for sample in self._dto.samples:
            frame = Frame()
            with open(
                f"{self._scene_path}/calibration/{sample.calibration_key}.json",
                "r",
            ) as f:
                calibration = CalibrationDTO.from_dict(json.load(f))

            extrinsics_by_sensor = dict(
                zip(
                    calibration.names,
                    map(
                        SensorExtrinsic.from_dto,
                        calibration.extrinsics,
                    ),
                )
            )

            intrinsics_by_sensor = dict(
                zip(
                    calibration.names,
                    map(
                        SensorIntrinsic.from_dto,
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
                    else Sensor(scene_path=self._scene_path, sensor_name=sensor_name)
                )
                sensor_frame = SensorFrame.from_dto(
                    sensor=sensor,
                    datum=data_row.datum,
                    extrinsic=extrinsics_by_sensor[sensor_name],
                    intrinsic=intrinsics_by_sensor[sensor_name],
                )
                sensor.add_sensor_frame(sensor_frame)
                frame.add_sensor(sensor_frame)

                sensors[sensor_name] = sensor

            self._add_frame(frame)

    @property
    def name(self) -> str:
        return self._dto.name

    @property
    def description(self) -> str:
        return self._dto.description

    @property
    def frames(self) -> List[Frame]:
        return self._frames

    @property
    def sensors(self) -> Dict[str, Sensor]:
        return self._sensors

    @staticmethod
    def from_dict(scene_data: Dict, dataset_path: str):
        scene = Scene(
            scene_dto=SceneDTO.from_dict(scene_data), dataset_path=dataset_path
        )
        return scene
