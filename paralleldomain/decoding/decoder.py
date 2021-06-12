import abc
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np

from paralleldomain.decoding.dgp_dto import (
    AnnotationsBoundingBox3DDTO,
    AnnotationsDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    DatasetDTO,
    SceneDTO,
)
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.sensor import Sensor, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class Decoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decode_scene_names(self) -> List[SceneName]:
        pass

    @abc.abstractmethod
    def decode_dataset_meta_data(self) -> DatasetMeta:
        pass

    @abc.abstractmethod
    def decode_scene_description(self, scene_name: SceneName) -> str:
        pass

    @abc.abstractmethod
    def decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        pass

    @abc.abstractmethod
    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        pass

    @abc.abstractmethod
    def decode_sensor(
        self,
        scene_name: SceneName,
        sensor_name: SensorName,
        sensor_frame_factory: Callable[[FrameId, SensorName], SensorFrame],
    ) -> Sensor:
        pass

    @abc.abstractmethod
    def decode_sensor_frame(self, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        pass

    @abc.abstractmethod
    def decode_available_sensor_names(self, scene_name: SceneName, frame_id: FrameId) -> List[SensorName]:
        pass
