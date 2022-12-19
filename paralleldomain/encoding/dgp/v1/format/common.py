import pickle
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, Union
from uuid import uuid4

import numpy as np

from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.encoding.pipeline_encoder import PipelineItem, ScenePipelineItem
from paralleldomain.model.annotation import BoundingBox2D, BoundingBoxes2D
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.mask import encode_2int16_as_rgba8

CUSTOM_FORMAT_KEY = "dgp_v1_data"
ANNOTATIONS_KEY = "annotations"
SENSOR_DATA_KEY = "sensor_data"
META_DATA_KEY = "metadata"
SCENE_DATA_KEY = "metadata"
CLASS_MAPS_KEY = "class_maps"
CAMERA_DATA_FOLDER = "camera_frames"
LIDAR_DATA_FOLDER = "lidar_frames"
ENCODED_SCENE_AGGREGATION_FOLDER_NAME = "scene_aggregation_tmp"


class CommonDGPV1FormatMixin:
    @staticmethod
    def _offset_timestamp(compare_datetime: datetime, reference_timestamp: datetime) -> float:
        diff = compare_datetime - reference_timestamp
        return diff.total_seconds()

    @staticmethod
    def _get_offset_timestamp_file_name(
        timestamp: datetime, scene_reference_timestamp: datetime, sim_offset: float
    ) -> str:
        offset = CommonDGPV1FormatMixin._offset_timestamp(
            compare_datetime=timestamp, reference_timestamp=scene_reference_timestamp
        )
        return f"{round((offset + sim_offset) * 100):018d}"

    @staticmethod
    def get_file_output_path(
        scene_output_path: Union[str, AnyPath],
        target_sensor_name: str,
        timestamp: datetime,
        scene_reference_timestamp: datetime,
        sim_offset: float,
        file_suffix: str,
        directory_name: str,
    ) -> AnyPath:
        scene_output_path = AnyPath(scene_output_path)
        file_name = CommonDGPV1FormatMixin._get_offset_timestamp_file_name(
            timestamp=timestamp, scene_reference_timestamp=scene_reference_timestamp, sim_offset=sim_offset
        )
        if file_suffix.startswith("."):
            file_suffix = file_suffix[1:]
        output_path = (
            scene_output_path
            / directory_name
            / target_sensor_name
            / f"{file_name}.{file_suffix}"
            # noqa: E501
        )
        return output_path

    @staticmethod
    def ensure_format_data_exists(pipeline_item: PipelineItem):
        if CUSTOM_FORMAT_KEY not in pipeline_item.custom_data:
            pipeline_item.custom_data[CUSTOM_FORMAT_KEY] = {
                ANNOTATIONS_KEY: dict(),
                SENSOR_DATA_KEY: dict(),
                META_DATA_KEY: dict(),
                CLASS_MAPS_KEY: dict(),
                SCENE_DATA_KEY: dict(),
            }


def encode_flow_vectors(vectors: np.ndarray) -> np.ndarray:
    height, width = vectors.shape[0:2]
    vectors = vectors / 2
    vectors = vectors / [width, height]
    vectors = vectors + 0.5
    vectors = vectors * 65535.0
    vectors = vectors.astype(np.int)
    return encode_2int16_as_rgba8(vectors)
