from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable, cast

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class CameraImageEncoderStep(EncoderStepHelper, EncoderStep):
    def __init__(
        self,
        fs_copy: bool,
        workers: int = 1,
        in_queue_size: int = 4,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.fs_copy = fs_copy

    def encode_camera_image_frame(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            output_path = self._get_dgpv1_file_output_path(
                sensor_frame=sensor_frame,
                input_dict=input_dict,
                file_suffix="png",
                directory_name=DirectoryName.RGB,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_file_path = sensor_frame.get_file_path(data_type=FilePathedDataType.Image)
            if self.fs_copy and image_file_path is not None:
                save_path = fsio.copy_file(source=image_file_path, target=output_path)
            else:
                save_path = fsio.write_png(obj=sensor_frame.image.rgba, path=output_path)

            if "sensor_data" not in input_dict:
                input_dict["sensor_data"] = dict()
            input_dict["sensor_data"]["rgb"] = save_path
        return input_dict

    def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = pypeln.thread.map(
            f=self.encode_camera_image_frame, stage=input_stage, workers=self.workers, maxsize=self.in_queue_size
        )
        return stage
