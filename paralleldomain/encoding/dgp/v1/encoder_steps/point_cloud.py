from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable, cast

import numpy as np
import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType, LidarSensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class PointCloudEncoderStep(EncoderStepHelper, EncoderStep):
    def __init__(
        self,
        fs_copy: bool,
        workers: int = 1,
        in_queue_size: int = 4,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.fs_copy = fs_copy

    @staticmethod
    def encode_point_cloud(sensor_frame: LidarSensorFrame[datetime], output_path: AnyPath) -> AnyPath:
        pc = sensor_frame.point_cloud
        pc_dtypes = [
            ("X", "<f4"),
            ("Y", "<f4"),
            ("Z", "<f4"),
            ("INTENSITY", "<f4"),
            ("R", "<f4"),
            ("G", "<f4"),
            ("B", "<f4"),
            ("RING_ID", "<u4"),
            ("TIMESTAMP", "<u8"),
        ]

        row_count = pc.length
        pc_data = np.empty(row_count, dtype=pc_dtypes)

        pc_data["X"] = pc.xyz[:, 0]
        pc_data["Y"] = pc.xyz[:, 1]
        pc_data["Z"] = pc.xyz[:, 2]
        pc_data["INTENSITY"] = pc.intensity[:, 0]
        pc_data["R"] = pc.rgb[:, 0]
        pc_data["G"] = pc.rgb[:, 1]
        pc_data["B"] = pc.rgb[:, 2]
        pc_data["RING_ID"] = pc.ring[:, 0]
        pc_data["TIMESTAMP"] = pc.ts[:, 0]

        return fsio.write_npz(obj={"data": pc_data}, path=output_path)

    def encode_point_cloud_frame(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            output_path = self._get_dgpv1_file_output_path(
                sensor_frame=sensor_frame,
                input_dict=input_dict,
                file_suffix="npz",
                directory_name=DirectoryName.POINT_CLOUD,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cloud_file_path = sensor_frame.get_file_path(data_type=FilePathedDataType.PointCloud)
            if self.fs_copy and cloud_file_path is not None:
                save_path = fsio.copy_file(source=cloud_file_path, target=output_path)
            else:
                save_path = self.encode_point_cloud(sensor_frame=sensor_frame, output_path=output_path)

            if "sensor_data" not in input_dict:
                input_dict["sensor_data"] = dict()
            input_dict["sensor_data"]["point_cloud"] = save_path
        return input_dict

    def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = pypeln.thread.map(
            f=self.encode_point_cloud_frame, stage=input_stage, workers=self.workers, maxsize=self.in_queue_size
        )
        return stage
