from datetime import datetime
from typing import Any, Dict, Iterable, Union

import numpy as np
import pypeln

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class CameraImageEncoderStep(EncoderStepHelper, EncoderStep):
    def __init__(
        self,
        fs_copy: bool,
        workers: int = 1,
        in_queue_size: int = 1,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.fs_copy = fs_copy

    def encode_camera_image_frame(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            image_file_path = sensor_frame.get_file_path(data_type=FilePathedDataType.Image)
            if self.fs_copy and image_file_path is not None:
                save_path = self.save_image(
                    image_or_path=image_file_path, sensor_frame=sensor_frame, input_dict=input_dict
                )
            else:
                save_path = self.save_image(
                    image_or_path=sensor_frame.image.rgba, sensor_frame=sensor_frame, input_dict=input_dict
                )

            if "sensor_data" not in input_dict:
                input_dict["sensor_data"] = dict()
            input_dict["sensor_data"]["rgb"] = save_path
        return input_dict

    def save_image(
        self,
        image_or_path: Union[np.ndarray, AnyPath],
        sensor_frame: CameraSensorFrame[datetime],
        input_dict: Dict[str, Any],
    ) -> AnyPath:
        output_path = self._get_dgpv1_file_output_path(
            sensor_frame=sensor_frame,
            input_dict=input_dict,
            file_suffix="png",
            directory_name=DirectoryName.RGB,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image_or_path, AnyPath):
            save_path = fsio.copy_file(source=image_or_path, target=output_path)
        else:
            save_path = fsio.write_png(obj=image_or_path, path=output_path)
        return save_path

    def apply(self, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = pypeln.thread.map(
            f=self.encode_camera_image_frame, stage=input_stage, workers=self.workers, maxsize=self.in_queue_size
        )
        return stage
