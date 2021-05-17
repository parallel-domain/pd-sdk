import abc
from typing import List

import numpy as np
from paralleldomain.dto import DatasetDTO, SceneDTO, CalibrationDTO, AnnotationsDTO, AnnotationsBoundingBox3DDTO, \
    CalibrationExtrinsicDTO, CalibrationIntrinsicDTO


class Decoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decode_dataset(self) -> DatasetDTO:
        pass

    @abc.abstractmethod
    def decode_scene(self, scene_name: str) -> SceneDTO:
        pass

    @abc.abstractmethod
    def decode_calibration(self, scene_name: str, calibration_key: str) -> CalibrationDTO:
        pass

    @abc.abstractmethod
    def decode_extrinsic_calibration(self, scene_name: str, calibration_key: str, sensor_name: str) \
            -> CalibrationExtrinsicDTO:
        pass

    @abc.abstractmethod
    def decode_intrinsic_calibration(self, scene_name: str, calibration_key: str, sensor_name: str) \
            -> CalibrationIntrinsicDTO:
        pass

    @abc.abstractmethod
    def decode_3d_bounding_boxes(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox3DDTO:
        pass

    @abc.abstractmethod
    def decode_point_cloud(self, scene_name: str, cloud_identifier: str, point_format: List[str]) -> np.ndarray:
        pass