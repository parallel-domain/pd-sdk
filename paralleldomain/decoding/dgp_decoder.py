import json
from typing import Union, List, cast, BinaryIO
import logging

import numpy as np
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.dto import DatasetDTO, DatasetMeta, SceneDTO, CalibrationDTO, AnnotationsBoundingBox3DDTO, \
    CalibrationExtrinsicDTO, CalibrationIntrinsicDTO
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPDecoder(Decoder):

    def __init__(self, dataset_path: Union[str, AnyPath]):
        self._dataset_path = AnyPath(dataset_path)

    def decode_dataset(self) -> DatasetDTO:
        dataset_cloud_path: AnyPath = AnyPath(self._dataset_path)
        scene_json_path: AnyPath = dataset_cloud_path / "scene_dataset.json"
        if not scene_json_path.exists():
            files_with_prefix = [name.name for name in dataset_cloud_path.iterdir() if "scene_dataset" in name.name]
            if len(files_with_prefix) == 0:
                logger.error(f"No scene_dataset.json or file starting with scene_dataset found under {dataset_cloud_path}!")
            scene_json_path: AnyPath = dataset_cloud_path / files_with_prefix[-1]

        with scene_json_path.open(mode="r") as f:
            scene_dataset = json.load(f)

        meta_data = DatasetMeta.from_dict(scene_dataset["metadata"])
        scene_names: List[str] = scene_dataset["scene_splits"]["0"]["filenames"]
        return DatasetDTO(meta_data=meta_data, scene_names=scene_names)

    def decode_scene(self, scene_name: str) -> SceneDTO:
        with (self._dataset_path / scene_name).open("r") as f:
            scene_data = json.load(f)
            scene_dto = SceneDTO.from_dict(scene_data)
            return scene_dto

    def decode_calibration(self, scene_name: str, calibration_key: str) -> CalibrationDTO:
        calibration_path = self._dataset_path / scene_name / "calibration" / f"{calibration_key}.json"
        with calibration_path.open("r") as f:
            cal_dict = json.load(f)
            return CalibrationDTO.from_dict(cal_dict)

    def decode_extrinsic_calibration(self, scene_name: str, calibration_key: str, sensor_name: str) \
            -> CalibrationExtrinsicDTO:
        calibration_dto = self.decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.extrinsics[index]

    def decode_intrinsic_calibration(self, scene_name: str, calibration_key: str, sensor_name: str) \
            -> CalibrationIntrinsicDTO:
        calibration_dto = self.decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.intrinsics[index]

    def decode_3d_bounding_boxes(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox3DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open("r") as f:
            return AnnotationsBoundingBox3DDTO.from_dict(json.load(f))

    def decode_point_cloud(self, scene_name: str, cloud_identifier: str, point_format: List[str]) -> np.ndarray:
        cloud_path = self._dataset_path / scene_name / cloud_identifier
        with cloud_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
        column_count = len(point_format)
        return np.array([f.tolist() for f in npz_data.f.data]).reshape(-1, column_count)