import abc
from typing import List

import numpy as np
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.decoding.dgp_dto import DatasetDTO, SceneDTO, CalibrationDTO, AnnotationsDTO, AnnotationsBoundingBox3DDTO, \
    CalibrationExtrinsicDTO, CalibrationIntrinsicDTO
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName, SceneName


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
    def decode_frame_ids(self, scene_name: SceneName) -> List[str]:
        pass

    @abc.abstractmethod
    def decode_sensor_names(self, scene_name: SceneName) -> List[str]:
        pass

    @abc.abstractmethod
    def decode_sensor_frame(self, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        pass

    @abc.abstractmethod
    def decode_available_sensor_names(self, scene_name: SceneName, frame_id: FrameId) -> List[SensorName]:
        pass