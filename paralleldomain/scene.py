from typing import Dict, List, Union
import ujson as json
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.utilities.any_path import AnyPath

from paralleldomain.dto import SceneDTO, CalibrationDTO, SceneSampleDTO, SceneDataDatum, AnnotationsDTO, SceneDataDTO
from paralleldomain.frame import Frame
from paralleldomain.sensor import Sensor, SensorFrame, SensorExtrinsic, SensorIntrinsic, SensorPose, SensorAnnotations, \
    LidarData


class Scene:
    def __init__(self, name: str, description: str, decoder: Decoder, samples: List[SceneSampleDTO],
                 # data: Dict[str, SceneDataDTO]
                 ):
        # self._dto = scene_dto
        # self._dataset_path = AnyPath(dataset_path)
        # self._scene_path = self._dataset_path / self.name
        self._samples = samples
        self._name = name
        self._description = description
        self._decoder = decoder
        self._frames: List[Frame] = []
        self._sensors: Dict[str, Sensor] = {}
        self._prepared_frames = False
    #
    # def _data_by_key(self) -> Dict[str, SceneDataDTO]:
    #     return {d.key: d for d in self._dto.data}

    def _add_frame(self, frame: Frame):
        self._frames.append(frame)
        self._sensors.update(
            {sf.sensor.name: sf.sensor for _, sf in frame.sensors.items()}
        )

    def _build_sensor_frame(self, sensor: Sensor, datum: SceneDataDatum, calibration_key: str) -> SensorFrame:
        if datum.image:
            file_name = datum.image.filename
            pose_loader = lambda: SensorPose.from_dto(dto=datum.image.pose)
            available_annotation_type_id_to_identifier = datum.image.annotations
            annotation_loader = lambda s: list()  # TODO
            data_loader = lambda: None
        else:
            file_name = datum.point_cloud.filename
            pose_loader = lambda: SensorPose.from_dto(dto=datum.point_cloud.pose)
            available_annotation_type_id_to_identifier = datum.point_cloud.annotations
            annotation_loader = lambda s: self._decoder.decode_3d_bounding_boxes(
                scene_name=self.name, annotation_identifier=s).annotations
            data_loader = lambda: LidarData(point_format=datum.point_cloud.point_format,
                                            load_data=lambda: self._decoder.decode_point_cloud(
                                                scene_name=self.name,
                                                cloud_identifier=datum.point_cloud.filename,
                                                point_format=datum.point_cloud.point_format))

        sensor_frame = SensorFrame(
            sensor=sensor,
            file_name=file_name,
            extrinsic_loader=lambda: SensorExtrinsic.from_dto(self._decoder.decode_extrinsic_calibration(
                scene_name=self.name,
                calibration_key=calibration_key,
                sensor_name=sensor.name)),
            intrinsic_loader=lambda: SensorIntrinsic.from_dto(self._decoder.decode_intrinsic_calibration(
                scene_name=self.name,
                calibration_key=calibration_key,
                sensor_name=sensor.name)),
            annotations_loader=lambda: SensorAnnotations(
                available_annotation_type_id_to_identifier=available_annotation_type_id_to_identifier,
                annotation_loader=annotation_loader),
            pose_loader=pose_loader,
            data_loader=data_loader
        )
        return sensor_frame

    def _prepare_frames(self):  # quick implementation, tbd better
        sensors: Dict[str, Sensor] = dict()
        data = self._data_by_key()
        for sample in self._samples:
            frame = Frame()
            # calibration_path = self._scene_path / "calibration" / f"{sample.calibration_key}.json"
            # with calibration_path.open("r") as f:
            #     cal_dict = json.load(f)
            #     calibration = CalibrationDTO.from_dict(cal_dict)
            #
            # extrinsics_by_sensor = dict(
            #     zip(
            #         calibration.names,
            #         map(
            #             SensorExtrinsic.from_dto,
            #             calibration.extrinsics,
            #         ),
            #     )
            # )
            #
            # intrinsics_by_sensor = dict(
            #     zip(
            #         calibration.names,
            #         map(
            #             SensorIntrinsic.from_dto,
            #             calibration.intrinsics,
            #         ),
            #     )
            # )

            for key in sample.datum_keys:
                data_row = data[key]
                sensor_name = data_row.id.name
                sensor = (
                    sensors[sensor_name]
                    if sensor_name in sensors.keys()
                    else Sensor(sensor_name=sensor_name)
                )
                sensor_frame = self._build_sensor_frame(sensor=sensor,
                                                        datum=data_row.datum,
                                                        calibration_key=sample.calibration_key)
                # sensor_frame = SensorFrame.from_dto(
                #     sensor=sensor,
                #     datum=data_row.datum,
                #     extrinsic=extrinsics_by_sensor[sensor_name],
                #     intrinsic=intrinsics_by_sensor[sensor_name],
                # )
                sensor.add_sensor_frame(sensor_frame)
                frame.add_sensor(sensor_frame)

                sensors[sensor_name] = sensor

            self._add_frame(frame)
        self._prepared_frames = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def frames(self) -> List[Frame]:
        if not self._prepared_frames:
            self._prepare_frames()
        return self._frames

    @property
    def sensors(self) -> Dict[str, Sensor]:
        if not self._prepared_frames:
            self._prepare_frames()
        return self._sensors

    # @staticmethod
    # def from_dto(dto: SceneDTO):
    #     scene = Scene(
    #         scene_dto=SceneDTO.from_dict(scene_data), dataset_path=dataset_path
    #     )
    #     return scene
    #
    # @staticmethod
    # def from_file(dataset_path: AnyPath, scene_name: str) -> "Scene":
    #     with (dataset_path / scene_name).open("r") as f:
    #         scene_data = json.load(f)
    #         return Scene.from_dict(scene_data=scene_data, dataset_path=dataset_path)
