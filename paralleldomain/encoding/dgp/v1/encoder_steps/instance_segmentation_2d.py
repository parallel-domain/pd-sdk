from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_step import EncoderStep
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.mask import encode_int32_as_rgb8


class InstanceSegmentation2DEncoderStep(EncoderStep):
    def __init__(
        self,
        fs_copy: bool,
        workers: int = 1,
        in_queue_size: int = 4,
    ):
        self.fs_copy = fs_copy
        self.in_queue_size = in_queue_size
        self.workers = workers

    def encode_instance_segmentation_2d(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if (
            sensor_frame is not None
            and AnnotationTypes.InstanceSegmentation2D in sensor_frame.available_annotation_types
        ):
            output_path = self._get_dgpv1_file_output_path(
                sensor_frame=sensor_frame,
                input_dict=input_dict,
                file_suffix="png",
                directory_name=DirectoryName.INSTANCE_SEGMENTATION_2D,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            instseg2d_file_path = sensor_frame.get_file_path(data_type=FilePathedDataType.InstanceSegmentation2D)
            if self.fs_copy and instseg2d_file_path is not None:
                save_path = fsio.copy_file(source=instseg2d_file_path, target=output_path)
            else:
                instseg2d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation2D)
                save_path = fsio.write_png(obj=instseg2d.rgb_encoded, path=output_path)

            if "annotations" not in input_dict:
                input_dict["annotations"] = dict()
            input_dict["annotations"][str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.InstanceSegmentation2D])] = save_path
        return input_dict

    def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = pypeln.thread.map(
            f=self.encode_instance_segmentation_2d, stage=input_stage, workers=self.workers, maxsize=self.in_queue_size
        )

        return stage
